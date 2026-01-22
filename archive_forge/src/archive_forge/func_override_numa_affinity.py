import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from shutil import which
from typing import List, Optional
import torch
from packaging.version import parse
def override_numa_affinity(local_process_index: int, verbose: Optional[bool]=None) -> None:
    """
    Overrides whatever NUMA affinity is set for the current process. This is very taxing and requires recalculating the
    affinity to set, ideally you should use `utils.environment.set_numa_affinity` instead.

    Args:
        local_process_index (int):
            The index of the current process on the current server.
        verbose (bool, *optional*):
            Whether to log out the assignment of each CPU. If `ACCELERATE_DEBUG_MODE` is enabled, will default to True.
    """
    if verbose is None:
        verbose = parse_flag_from_env('ACCELERATE_DEBUG_MODE', False)
    if torch.cuda.is_available():
        from accelerate.utils import is_pynvml_available
        if not is_pynvml_available():
            raise ImportError('To set CPU affinity on CUDA GPUs the `pynvml` package must be available. (`pip install pynvml`)')
        import pynvml as nvml
        nvml.nvmlInit()
        num_elements = math.ceil(os.cpu_count() / 64)
        handle = nvml.nvmlDeviceGetHandleByIndex(local_process_index)
        affinity_string = ''
        for j in nvml.nvmlDeviceGetCpuAffinity(handle, num_elements):
            affinity_string = f'{j:064b}{affinity_string}'
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()
        affinity_to_set = [i for i, e in enumerate(affinity_list) if e != 0]
        os.sched_setaffinity(0, affinity_to_set)
        if verbose:
            cpu_cores = os.sched_getaffinity(0)
            logger.info(f'Assigning {len(cpu_cores)} cpu cores to process {local_process_index}: {cpu_cores}')