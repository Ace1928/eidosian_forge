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
def clean_device_map(device_map: Dict[str, Union[int, str, torch.device]], module_name: str=''):
    """
    Cleans a device_map by grouping all submodules that go on the same device together.
    """
    prefix = '' if module_name == '' else f'{module_name}.'
    values = [v for k, v in device_map.items() if k.startswith(prefix)]
    if len(set(values)) == 1 and len(values) > 1:
        for k in [k for k in device_map if k.startswith(prefix)]:
            del device_map[k]
        device_map[module_name] = values[0]
    children_modules = [k for k in device_map.keys() if k.startswith(prefix) and len(k) > len(module_name)]
    idx = len(module_name.split('.')) + 1 if len(module_name) > 0 else 1
    children_modules = set(('.'.join(k.split('.')[:idx]) for k in children_modules))
    for child in children_modules:
        clean_device_map(device_map, module_name=child)
    return device_map