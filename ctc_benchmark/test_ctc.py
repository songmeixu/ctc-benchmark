import argparse
import os
import random
import shutil
import time
import warnings

import torch
from ctc_benchmark.utils.log import logger
from ctc_benchmark.utils.benchmark import TestBencher


ctc_candidates = ['k2.CtcLoss', 'warpctc', 'torch.nn.CTCLoss', 'tf.nn.ctc_loss']


def ctc_benchmark():

    for c in ctc_candidates:
        TestBencher.run_benchmark(c)


def main():
    parser = argparse.ArgumentParser(description='CTC Benchmarks')

    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='for seed init')
    parser.add_argument('--output_dir',
                        default="results",
                        type=str,
                        help='output directory for logs, stats and figures')


    args = parser.parse_args()

    args.nprocs = torch.cuda.device_count()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.local_rank, args.nprocs, args)

    ctc_benchmark()


if __name__ == '__main__':
    main()