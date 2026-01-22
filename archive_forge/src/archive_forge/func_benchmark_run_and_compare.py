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
def benchmark_run_and_compare(benchmark_fn, cases: List[Dict[str, Any]], compare: List[str], omit_baselines: bool=False, fail_if_regression: bool=False, quiet: bool=False, optimized_label: str='optimized', *, min_run_time: float=2.0, atol_s: float=3e-05, rtol: float=0.05) -> None:
    SKIP_VANILLA_TASKS_IF_ALREADY_DONE = True
    results_compare_to = []
    results = []
    store_results_folder = os.path.expanduser(os.path.join(os.environ.get('XFORMERS_BENCHMARKS_CACHE', os.path.join('~', '.cache', 'xformers', 'benchmarks')), get_func_name(benchmark_fn)))
    try:
        env = torch.cuda.get_device_name(torch.cuda.current_device()).replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_')
    except (RuntimeError, AssertionError):
        env = 'cpu'
    assert '.' not in optimized_label, f'label=`{optimized_label}` should not contain dots'
    assert '.' not in env, f'env=`{env}` should not contain dots'
    os.makedirs(store_results_folder, exist_ok=True)
    skip_vanilla_tasks = set()
    for cmp_name in compare:
        name_with_env = cmp_name if '.' in cmp_name else f'{cmp_name}.*'
        for filename in glob.glob(os.path.join(store_results_folder, f'{name_with_env}.csv')):
            loaded = _benchmark_results_from_csv(filename)
            for m, r in loaded:
                if m.get(META_ALGORITHM) is not None:
                    m[META_ALGORITHM] = m[META_ALGORITHM].partition('@')[0]
                if r.task_spec.env == env and SKIP_VANILLA_TASKS_IF_ALREADY_DONE:
                    skip_vanilla_tasks.add((r.task_spec.sub_label, r.task_spec.num_threads))
            results_compare_to += loaded
    if not quiet:
        pbar = tqdm.tqdm(cases, leave=False)
        cases = pbar
    for case in cases:
        if quiet:
            print(str(case))
        else:
            pbar.write(f'====== {str(case)} ======')
        try:
            benchmarks_generator = benchmark_fn(**case)
        except NotImplementedError:
            continue
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            if not quiet:
                pbar.write('Skipped (OOM)')
            continue
        name = None
        try:
            for benchmark_object in benchmarks_generator:
                is_optimized = benchmark_object._task_spec.description not in BASELINE_DESCRIPTIONS
                metadata = {}
                if is_optimized:
                    metadata[META_ALGORITHM] = benchmark_object._task_spec.description
                    benchmark_object._task_spec = replace(benchmark_object._task_spec, description=optimized_label)
                elif omit_baselines or (benchmark_object._task_spec.sub_label, benchmark_object._task_spec.num_threads) in skip_vanilla_tasks:
                    continue
                memory = math.inf
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    mem_begin = torch.cuda.max_memory_allocated() / 2 ** 20
                    benchmark_object._task_spec = replace(benchmark_object._task_spec, env=env)
                    measurement = benchmark_object.blocked_autorange(min_run_time=min_run_time)
                    torch.cuda.synchronize()
                    results.append((metadata, measurement))
                    name = measurement.task_spec.description
                    memory = torch.cuda.max_memory_allocated() / 2 ** 20 - mem_begin
                    measurement.mem_use = memory
                except RuntimeError as e:
                    if not _is_oom_error(e):
                        raise
                    if not quiet:
                        pbar.write('Skipped (OOM)')
                finally:
                    del benchmark_object
                if not quiet:
                    pbar.write(f'{name}: memory used: {memory} MB')
        except RuntimeError as e:
            if not _is_oom_error(e):
                raise
            if not quiet:
                pbar.write('Skipped (OOM)')
        if name is not None and (not quiet):

            def matches_current(r):
                return r[1].task_spec.sub_label == results[-1][1].task_spec.sub_label and r[1].task_spec.label == results[-1][1].task_spec.label
            pbar.write(str(benchmark.Compare(_finalize_results(list(filter(matches_current, results)) + list(filter(matches_current, results_compare_to))))))
    results_for_print = _finalize_results(results + results_compare_to)
    benchmark.Compare(results_for_print).print()
    _render_bar_plot(results_for_print, store_results_folder)
    if results and optimized_label is not None:
        write_to_path = os.path.join(store_results_folder, f'{optimized_label}.{env}.csv')
        _benchmark_results_to_csv(write_to_path, results)
        print(f'Saved results to {write_to_path}')
    if fail_if_regression:
        _fail_if_regressions(results, reference=results_compare_to, atol_s=atol_s, rtol=rtol)