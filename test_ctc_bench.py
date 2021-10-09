import os
import time
import pytest
import torch
import k2

from ctc_benchmark.utils.log import logger


# ctc_candidates = [
#     # 'k2.ctc_Loss',
#     # 'warpctc',
#     torch.nn.functional.ctc_loss,
#     # torch.nn.functional.ctc_loss,
#     # 'tf.nn.ctc_loss'
# ]


@pytest.mark.benchmark(
    group='ctc',
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
)
class TestCTCBench:

    def prepare_inputs(self) -> None:
        """
        random initialize ctc batch inputs:
            - log_probs: Tensor
            - targets: Tensor,
            - input_lengths: Tensor
            - target_lengths: Tensor
        """
        # batch_size = 32
        self.input_length = 1000
        self.vocab_size = 100 # include blank (= 0, by default)
        self.batch_size = 32
        target_length = 50

        # [T, N, C]
        self.log_probs = torch.randn(
            self.input_length, self.batch_size, self.vocab_size
        ).log_softmax(2).requires_grad_()
        self.targets = torch.randint(
            low=1, high=self.vocab_size - 1,
            size=(self.batch_size, target_length), dtype=torch.long
        )
        self.input_lengths = self.batch_size * [self.input_length]
        self.target_lengths = self.batch_size * [target_length]

    def convert_inputs_to_k2(self):
        # convert inputs as k2 requires:
        #   log_probs -> DenseFsaVec,
        #   targets -> Fsa,
        supervision_segments = torch.tensor([[0, 0, self.input_length]],
                                            dtype=torch.int32)
        for n in range(1, self.batch_size):
            supervision_segments = torch.cat((
                supervision_segments,
                torch.tensor([[n, 0, self.input_length]], dtype=torch.int32)
            ))
        # [T, N, C] => [N, T, C]
        k2_log_probs = self.log_probs.permute(1, 0, 2)
        dense_fsa_vec = k2.DenseFsaVec(
            k2_log_probs,
            supervision_segments
        )

        ctc_topo = k2.ctc_topo(self.vocab_size - 1)
        linear_fsa = k2.linear_fsa(self.targets.tolist())
        decoding_graph = k2.compose(
            ctc_topo, linear_fsa
        )

        return dense_fsa_vec, decoding_graph

    def test_k2_forward(self, benchmark):
        self.prepare_inputs()
        dense_fsa_vec, decoding_graph = self.convert_inputs_to_k2()

        k2_loss = benchmark(
            k2.ctc_loss,
            decoding_graph,
            dense_fsa_vec,
            reduction='sum',
            target_lengths=self.target_lengths
        )

    def test_k2_backward(self, benchmark):
        self.prepare_inputs()
        dense_fsa_vec, decoding_graph = self.convert_inputs_to_k2()

        k2_loss = k2.ctc_loss(
            decoding_graph,
            dense_fsa_vec,
            reduction='sum',
            target_lengths=self.target_lengths
        )

        grad_out = torch.randn_like(k2_loss)
        benchmark(
            torch.autograd.grad,
            k2_loss, self.log_probs, grad_out.cuda(),
            retain_graph=True
        )

    def test_torch_forward(self, benchmark, use_cudnn: bool = False):
        self.prepare_inputs()

        with torch.backends.cudnn.flags(enabled=use_cudnn):
            forward_res = benchmark(
                torch.nn.functional.ctc_loss,
                self.log_probs.cuda(), self.targets.cuda(),
                self.input_lengths, self.target_lengths,
                reduction='sum', zero_infinity=True
            )

    def test_torch_backward(self, benchmark, use_cudnn: bool = False):
        self.prepare_inputs()

        with torch.backends.cudnn.flags(enabled=use_cudnn):
            forward_res = torch.nn.functional.ctc_loss(
                self.log_probs.cuda(), self.targets.cuda(),
                self.input_lengths, self.target_lengths,
                reduction='sum', zero_infinity=True
            )

            grad_out = torch.randn_like(forward_res)
            benchmark(
                torch.autograd.grad,
                forward_res, self.log_probs, grad_out.cuda(),
                retain_graph=True
            )

    def test_torch_forward_with_cudnn(self, benchmark):
        self.test_torch_forward(benchmark, use_cudnn=True)

    def test_torch_backward_with_cudnn(self, benchmark):
        self.test_torch_backward(benchmark, use_cudnn=True)

    def test_warpctc_forward(self, benchmark):
        return

    def test_warpctc_backward(self, benchmark):
        return

    def test_tf_forward(self, benchmark):
        return

    def test_tf_backward(self, benchmark):
        return
