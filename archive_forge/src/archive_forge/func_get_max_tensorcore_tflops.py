import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from ._C.libtriton.triton import runtime
def get_max_tensorcore_tflops(dtype, clock_rate, backend=None, device=None):
    import torch
    from .runtime import driver
    if not backend:
        backend = runtime.backend.CUDA
    if not device:
        device = torch.cuda.current_device()
    num_subcores = driver.utils.get_device_properties(device)['multiprocessor_count'] * 4
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 8:
        assert dtype == torch.float16
        ops_per_sub_core = 256
    elif dtype in [torch.float32, torch.int32]:
        ops_per_sub_core = 256
    elif dtype in [torch.float16, torch.bfloat16, torch.int16]:
        ops_per_sub_core = 512
    elif dtype in [torch.int8, tl.float8e4nv, tl.float8e4b15, tl.float8e5]:
        ops_per_sub_core = 1024
    else:
        raise RuntimeError('dtype not supported')
    tflops = num_subcores * clock_rate * ops_per_sub_core * 1e-09
    return tflops