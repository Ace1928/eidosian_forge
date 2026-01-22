import contextlib
import importlib
import os
import sys
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, cast, List, Optional, Tuple, Union
import torch
import torch._C
from torch.types import Device
from .. import device as _device
from .._utils import classproperty
from ._utils import _dummy_type, _get_device_index
from .graphs import (
from .streams import Event, ExternalStream, Stream
from .memory import *  # noqa: F403
from .random import *  # noqa: F403
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from . import amp, jiterator, nvtx, profiler, sparse
def is_bf16_supported():
    """Return a bool indicating if the current CUDA/ROCm device supports dtype bfloat16."""
    if torch.version.hip:
        return True
    cu_vers = torch.version.cuda
    if cu_vers is not None:
        cuda_maj_decide = int(cu_vers.split('.')[0]) >= 11
    else:
        cuda_maj_decide = False
    return torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8 and cuda_maj_decide