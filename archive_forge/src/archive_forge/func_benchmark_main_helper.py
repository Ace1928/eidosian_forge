import argparse
import contextlib
import copy
import csv
import functools
import glob
import itertools
import logging
import math
import os
import tempfile
from collections import defaultdict, namedtuple
from dataclasses import replace
from typing import Any, Dict, Generator, Iterator, List, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tqdm
from torch.utils import benchmark
def benchmark_main_helper(benchmark_fn, cases: List[Dict[str, Any]], **kwargs) -> None:
    """
    Helper function to run benchmarks.
    Supports loading previous results for comparison, and saving current results to file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', default=None, type=str, help='Only benchmark this function')
    parser.add_argument('--label', default=None, type=str, help='Store results to a file')
    parser.add_argument('--fail_if_regression', action='store_true', help='Enabled in CI to check against performance regressions')
    parser.add_argument('--compare', default=None, type=str, help='Compare to previously stored benchmarks (coma separated)')
    parser.add_argument('--omit-baselines', action='store_true', help='Do not run the (potentially slow) baselines')
    parser.add_argument('--quiet', action='store_true', help='Skip intermediate results and progress bar')
    args = parser.parse_args()
    if args.fn is not None and args.fn != get_func_name(benchmark_fn):
        print(f'Skipping benchmark "{get_func_name(benchmark_fn)}"')
        return
    benchmark_run_and_compare(benchmark_fn=benchmark_fn, cases=cases, optimized_label='optimized' if args.label is None else args.label, fail_if_regression=args.fail_if_regression, compare=args.compare.split(',') if args.compare is not None else [], quiet=args.quiet, omit_baselines=args.omit_baselines, **kwargs)