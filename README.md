
<h3 align="center">CTC Benchmark</h3>

---

<p align="center"> What's the best CTC implementation?
    <br>
</p>

## About <a name = "about"></a>
This project aims to benchmark the performance of some ctc algorithm implementations.

Currently include:
- [torch.nn.CTCLoss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
- [tf.nn.ctc_loss](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss)
- [k2.CtcLoss](https://k2-fsa.github.io/k2/python_api/api.html#ctc-loss)

## Getting Started <a name = "getting_started"></a>
```bash
pip install -r requirements.txt
```

## Running the benchmark tests <a name = "tests"></a>

```bash
# run ctc:forward (get ctc-loss) tests
pytest -k "forward" --benchmark-columns='mean' .
# run ctc:backward (calculate gradients) tests
pytest -k "backward" --benchmark-columns='mean' .
```

### Results
- ctc forward benchmark results

![](results/forward.png "forward benchmark results")
- ctc backward benchmark results

![](results/backward.png "backward benchmark results")

### Versions
- TF: v2.6.0
- torch: v1.8.2
- k2: v1.9

### Explainations
- TF test in eager execution mode, this may introduce the overhead
- K2 test at python API level as others, which include `py->c++` call cost. K2 uses pybind11 as the binding framework, which may introduce the overhead.
- CTC implementation of CUDNN is the backend of `torch_with_cudnn` and `tf_with_gpu`

## Todo
- [ ] Remove TF eager mode overhead from benchmark results.
- [ ] Remove K2 pybind11 overhead between python API -> c++ API from benchmark results, through call c++ API directly.
- [ ] Add [GTN](https://github.com/facebookresearch/gtn) ctc alternative.
- [ ] Add comparations about more aspects of those CTC implementations.

## Built Using <a name = "built_using"></a>
- [Pytest](https://github.com/pytest-dev/pytest) - Python Test Framework
- [Pytest-benchmark](https://github.com/ionelmc/pytest-benchmark/) - Benchmark Framework within Pytest
