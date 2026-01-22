import functools
from typing import Dict, Optional, Tuple, Union
import torch
from torch.cuda import _CudaDeviceProperties
def current_device() -> int:
    if _compile_worker_current_device is not None:
        return _compile_worker_current_device
    return torch.cuda.current_device()