import os
import time
import pytest
import torch
import tensorflow as tf
import k2


@pytest.mark.benchmark(
    group='ctc',
    warmup=True,
    warmup_iterations=3,
    disable_gc=True,
    timer=time.perf_counter,
    min_rounds=10,
    # min_time=1,
    # max_time=10,
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
        torch.manual_seed(1987)

        # batch_size = 32
        self.input_length = 150
        self.vocab_size = 28 # include blank (= 0, by default)
        self.batch_size = 32
        target_length = 40

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

    def convert_inputs_to_tf(self):
        labels = self.targets.detach().numpy()
        labels = tf.convert_to_tensor(labels)

        logits = self.log_probs.detach().numpy()
        logits = tf.convert_to_tensor(logits)

        return labels, logits

    def test_k2_forward(self, benchmark):
        self.prepare_inputs()
        dense_fsa_vec, decoding_graph = self.convert_inputs_to_k2()

        benchmark(
            k2.ctc_loss,
            decoding_graph,
            dense_fsa_vec,
            reduction='sum',
            target_lengths=self.target_lengths
        )

    def test_k2_backward(self, benchmark):
        self.prepare_inputs()
        dense_fsa_vec, decoding_graph = self.convert_inputs_to_k2()

        ctc_loss = k2.ctc_loss(
            decoding_graph,
            dense_fsa_vec,
            reduction='sum',
            target_lengths=self.target_lengths
        )

        grad_out = torch.randn_like(ctc_loss)
        benchmark(
            torch.autograd.grad,
            ctc_loss, self.log_probs, grad_out.cuda(),
            retain_graph=True
        )

    def test_torch_forward(self, benchmark, use_cudnn: bool = False):
        self.prepare_inputs()

        with torch.backends.cudnn.flags(enabled=use_cudnn):
            benchmark(
                torch.nn.functional.ctc_loss,
                self.log_probs.cuda(), self.targets.cuda(),
                self.input_lengths, self.target_lengths,
                reduction='sum', zero_infinity=True
            )

    def test_torch_backward(self, benchmark, use_cudnn: bool = False):
        self.prepare_inputs()

        with torch.backends.cudnn.flags(enabled=use_cudnn):
            ctc_loss = torch.nn.functional.ctc_loss(
                self.log_probs.cuda(), self.targets.cuda(),
                self.input_lengths, self.target_lengths,
                reduction='sum', zero_infinity=True
            )

            grad_out = torch.randn_like(ctc_loss)
            benchmark(
                torch.autograd.grad,
                ctc_loss, self.log_probs, grad_out.cuda(),
                retain_graph=True
            )

    def test_torch_forward_with_cudnn(self, benchmark):
        self.test_torch_forward(benchmark, use_cudnn=True)

    def test_torch_backward_with_cudnn(self, benchmark):
        self.test_torch_backward(benchmark, use_cudnn=True)

    def test_tf_forward(self, benchmark, device='gpu'):
        self.prepare_inputs()
        labels, logits = self.convert_inputs_to_tf()

        with tf.device(f"/{device}:0"):
            benchmark(
                tf.nn.ctc_loss,
                labels=labels,
                logits=logits,
                logit_length=self.input_lengths,
                label_length=self.target_lengths
                )

    def test_tf_backward(self, benchmark, device='gpu'):
        self.prepare_inputs()
        labels, logits = self.convert_inputs_to_tf()

        with tf.device(f"/{device}:0"):
            with tf.GradientTape(persistent=True) as t:
                t.watch(logits)
                ctc_loss = tf.nn.ctc_loss(
                    labels=labels,
                    logits=logits,
                    logit_length=self.input_lengths,
                    label_length=self.target_lengths
                )

            benchmark(
                t.gradient,
                ctc_loss, [logits]
            )

    def test_tf_forward_cpu(self, benchmark):
        self.test_tf_forward(benchmark, device='cpu')

    def test_tf_backward_cpu(self, benchmark):
        self.test_tf_backward(benchmark, device='cpu')
