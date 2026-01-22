import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from ._C.libtriton.triton import runtime
def cuda_memcheck(**target_kwargs):

    def decorator(test_fn):

        @functools.wraps(test_fn)
        def wrapper(*args, **kwargs):
            import psutil
            ppid_name = psutil.Process(os.getppid()).name()
            run_cuda_memcheck = target_kwargs.items() <= kwargs.items()
            if run_cuda_memcheck and ppid_name != 'cuda-memcheck':
                path = os.path.realpath(test_fn.__globals__['__file__'])
                env = {'PATH': os.environ['PATH'], 'PYTORCH_NO_CUDA_MEMORY_CACHING': '1'}
                assert 'request' in kwargs, "memcheck'ed test must have a (possibly unused) `request` fixture"
                test_id = kwargs['request'].node.callspec.id
                cmd = f'{path}::{test_fn.__name__}[{test_id}]'
                out = subprocess.run(['cuda-memcheck', 'pytest', '-vs', cmd], capture_output=True, env=env)
                assert out.returncode == 0, 'cuda-memcheck returned an error: bounds checking failed'
                assert 'ERROR SUMMARY: 0 errors' in str(out.stdout)
            else:
                test_fn(*args, **kwargs)
        return wrapper
    return decorator