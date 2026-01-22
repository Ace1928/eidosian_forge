import contextlib
import gc
import importlib
import inspect
import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Tuple, Union
import packaging
import torch
import torch.nn as nn
from ..state import AcceleratorState
from .constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from .dataclasses import AutocastKwargs, CustomDtype, DistributedType
from .imports import is_mps_available, is_npu_available, is_peft_available, is_torch_xla_available, is_xpu_available
from .offload import load_offloaded_weight, offload_weight, save_offload_index
from .tqdm import is_tqdm_available, tqdm
from .versions import compare_versions
from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
def get_max_memory(max_memory: Optional[Dict[Union[int, str], Union[int, str]]]=None):
    """
    Get the maximum memory available if nothing is passed, converts string to int otherwise.
    """
    import psutil
    if max_memory is None:
        if not (torch.cuda.is_available() or is_npu_available() or is_xpu_available()):
            max_memory = {}
        elif is_npu_available():
            for i in range(torch.npu.device_count()):
                _ = torch.tensor(0, device=torch.device('npu', i))
            max_memory = {i: torch.npu.mem_get_info(i)[0] for i in range(torch.npu.device_count())}
        elif is_xpu_available():
            for i in range(torch.xpu.device_count()):
                _ = torch.tensor(0, device=torch.device('xpu', i))
            max_memory = {i: torch.xpu.max_memory_allocated(i) for i in range(torch.xpu.device_count())}
        else:
            for i in range(torch.cuda.device_count()):
                _ = torch.tensor([0], device=i)
            max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}
        if is_mps_available():
            max_memory['mps'] = psutil.virtual_memory().available
        else:
            max_memory['cpu'] = psutil.virtual_memory().available
        return max_memory
    for key in max_memory:
        if isinstance(max_memory[key], str):
            max_memory[key] = convert_file_size_to_int(max_memory[key])
    gpu_devices = [k for k in max_memory.keys() if isinstance(k, int)]
    gpu_devices.sort()
    if is_npu_available():
        num_devices = torch.npu.device_count()
    elif is_xpu_available():
        num_devices = torch.xpu.device_count()
    else:
        num_devices = torch.cuda.device_count()
    for device in gpu_devices:
        if device >= num_devices or device < 0:
            logger.warning(f'Device {device} is not available, available devices are {list(range(num_devices))}')
    all_devices = gpu_devices + [k for k in ['mps', 'cpu', 'disk'] if k in max_memory.keys()]
    for k in max_memory.keys():
        if k not in all_devices:
            raise ValueError(f"Device {k} is not recognized, available devices are integers(for GPU/XPU), 'mps', 'cpu' and 'disk'")
    max_memory = {k: max_memory[k] for k in all_devices}
    return max_memory