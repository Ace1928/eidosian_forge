import contextlib
import time
import os
import json
import torch
from torch.profiler import profile, ProfilerActivity
def benchmark_utilization(f, input, trace_folder, optimize_ctx=None, trace_file_name='tmp_chrome_trace', num_runs=1):
    """
    Benchmark the GPU Utilization and percent of time spent on matmul and convolution operations of
    running f(input, **kwargs_for_f) with [optimize_ctx] [num_runs] times.
    It will produce a chrome trace file in trace_folder/trace_file_name.json

    Example:

    ```
    def f(a):
        return a.sum()
    a = torch.rand(2**20, device="cuda")
    utilization, mm_conv_utilization = benchmark_utilization(f, a, "tmp", trace_file_name = "tmp_chrome_trace")
    ```

    Args:
        f: function to benchmark

        input: input to :attr:`f`

        trace_folder: name of the folder to store the chrome trace

        optimize_ctx: the context in which f will run

        trace_file_name: name of the dumped chrome trace file, default to "tmp_chrome_trace"

        num_runs: number of times to run f, excluding the warm-up runs, default to 1.

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)

    """
    isExist = os.path.exists(trace_folder)
    if not isExist:
        os.makedirs(trace_folder)
        print('create folder ' + trace_folder)
    if optimize_ctx is None:
        optimize_ctx = contextlib.nullcontext()
    chrome_trace_file_name = os.path.join(trace_folder, trace_file_name + '.json')
    total_length = dump_chrome_trace(f, input, chrome_trace_file_name, optimize_ctx, [ProfilerActivity.CUDA], num_runs=num_runs, devices='cuda')
    utilization, mm_conv_utilization = compute_utilization(chrome_trace_file_name, total_length)
    return (utilization, mm_conv_utilization)