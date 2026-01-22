import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from ._C.libtriton.triton import runtime
def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode='mean'):
    assert return_mode in ['min', 'max', 'mean', 'median']
    import torch
    '\n    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with\n    the 20-th and 80-th performance percentile.\n\n    :param fn: Function to benchmark\n    :type fn: Callable\n    :param warmup: Warmup time (in ms)\n    :type warmup: int\n    :param rep: Repetition time (in ms)\n    :type rep: int\n    :param grad_to_none: Reset the gradient of the provided tensor to None\n    :type grad_to_none: torch.tensor, optional\n    :param quantiles: Performance percentile to return in addition to the median.\n    :type quantiles: list[float]\n    :param fast_flush: Use faster kernel to flush L2 between measurements\n    :type fast_flush: bool\n    '
    fn()
    torch.cuda.synchronize()
    if fast_flush:
        cache = torch.empty(int(256000000.0 // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(256000000.0), dtype=torch.int8, device='cuda')
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    for _ in range(n_warmup):
        fn()
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        cache.zero_()
        start_event[i].record()
        fn()
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()