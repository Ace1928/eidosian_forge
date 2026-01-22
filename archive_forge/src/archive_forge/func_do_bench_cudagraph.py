import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from ._C.libtriton.triton import runtime
def do_bench_cudagraph(fn, rep=20, grad_to_none=None):
    import torch
    '\n    Benchmark the runtime of the provided function.\n\n    :param fn: Function to benchmark\n    :type fn: Callable\n    :param rep: Repetition time (in ms)\n    :type rep: int\n    :param grad_to_none: Reset the gradient of the provided tensor to None\n    :type grad_to_none: torch.tensor, optional\n    '
    if torch.cuda.current_stream() == torch.cuda.default_stream():
        raise RuntimeError('Cannot capture graph in default stream. Please use side stream in benchmark code.')
    fn()
    if grad_to_none is not None:
        for x in grad_to_none:
            x.detach_()
            x.requires_grad_(True)
            x.grad = None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    g.replay()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event)
    n_repeat = max(1, int(rep / estimate_ms))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_repeat):
            if grad_to_none is not None:
                for x in grad_to_none:
                    x.grad = None
            fn()
    torch.cuda.synchronize()
    ret = []
    n_retries = 10
    for i in range(n_retries):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        g.replay()
        end_event.record()
        torch.cuda.synchronize()
        ret += [start_event.elapsed_time(end_event) / n_repeat]
    return torch.mean(torch.tensor(ret)).item()