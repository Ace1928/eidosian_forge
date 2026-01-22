import argparse
import itertools as it
import multiprocessing
import multiprocessing.dummy
import os
import pickle
import queue
import subprocess
import tempfile
import textwrap
import numpy as np
import torch
from torch.utils.benchmark.op_fuzzers import unary
from torch.utils.benchmark import Timer, Measurement
from typing import Dict, Tuple, List
def process_results(results, test_variance):
    paired_results: Dict[Tuple[str, str, int, bool, int], List] = {}
    for (seed, use_gpu), result_batch in results:
        for r in result_batch:
            key = (r.label, r.description, r.num_threads, use_gpu, seed)
            paired_results.setdefault(key, [[], []])
            index = 0 if r.env.startswith('ref') else 1
            paired_results[key][index].append(r)
    paired_results = {key: [merge(r_ref_list), merge(r_pr_list)] for key, (r_ref_list, r_pr_list) in paired_results.items()}
    flagged_for_removal = set()
    for key, (r_ref, r_pr) in paired_results.items():
        if any((r is None or r.has_warnings for r in (r_ref, r_pr))):
            flagged_for_removal.add(key)
    paired_results = {k: v for k, v in paired_results.items() if k not in flagged_for_removal}
    print(f'{len(flagged_for_removal)} samples were culled, {len(paired_results)} remain')
    gpu_results = [(k, v) for k, v in paired_results.items() if k[3]]
    cpu_results = [(k, v) for k, v in paired_results.items() if not k[3]]
    if cpu_results:
        construct_table(cpu_results, 'CPU', test_variance)
    if gpu_results:
        construct_table(gpu_results, 'GPU', test_variance)