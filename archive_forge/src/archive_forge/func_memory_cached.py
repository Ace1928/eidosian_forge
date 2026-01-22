import collections
import contextlib
import ctypes
import pickle
import sys
import warnings
from inspect import signature
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import _C
from torch.types import Device
from . import _get_device_index, _get_nvml_device_index, _lazy_init, is_initialized
from ._memory_viz import memory as _memory, segments as _segments
from ._utils import _dummy_type
def memory_cached(device: Union[Device, int]=None) -> int:
    """Deprecated; see :func:`~torch.cuda.memory_reserved`."""
    warnings.warn('torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved', FutureWarning)
    return memory_reserved(device=device)