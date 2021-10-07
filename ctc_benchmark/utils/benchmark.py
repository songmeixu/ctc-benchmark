import time
import pytest

from ctc_benchmark.utils.log import logger


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
    group='hub',
)
class TestBencher:
    def run_benchmark() -> None:
        logger.info("Preparing ctc input data...")
        logger.info("Preparing ctc output data...")
