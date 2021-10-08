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

    def prepare_inputs(self, ctc) -> None:
        """
        random initialize ctc batch inputs:
            - log_probs: Tensor
            - targets: Tensor,
            - input_lengths: Tensor
            - target_lengths: Tensor
        """
        # batch_size = 32
        input_length = 1000
        vocab_size = 100 # include blank (= 0, by default)
        batch_size = 32
        target_length = 50

        # [T, N, C]
        self.log_probs = torch.randn(
            input_length, batch_size, vocab_size
        ).log_softmax(2).requires_grad_()
        self.targets = torch.randint(
            low=1, high=vocab_size - 1,
            size=(batch_size, target_length), dtype=torch.long
        )
        self.input_lengths = batch_size * [input_length]
        self.target_lengths = batch_size * [target_length]

        if ctc is k2.ctc_loss:
            # convert inputs as k2 requires:
            #   log_probs -> DenseFsaVec,
            #   targets -> Fsa,
            print()

    def test_k2_forward(self, benchmark):
        return

    def test_k2_backward(self, benchmark):
        return

    def test_torch_forward(self, benchmark, use_cudnn: bool = False):
        for ctc in ctc_candidates:
            self.prepare_inputs(ctc)

            with torch.backends.cudnn.flags(enabled=use_cudnn):
                forward_res = benchmark(
                    ctc,
                    self.log_probs.cuda(), self.targets.cuda(),
                    self.input_lengths, self.target_lengths,
                    reduction='sum', zero_infinity=True
                )

    def test_torch_backward(self, benchmark, use_cudnn: bool = False):
        for ctc in ctc_candidates:
            self.prepare_inputs(ctc)

            with torch.backends.cudnn.flags(enabled=use_cudnn):
                forward_res = ctc(
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
