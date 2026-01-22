import functools
from typing import Dict, Optional, Tuple, Union
import torch
from torch.cuda import _CudaDeviceProperties
def set_compiler_worker_current_device(device: int) -> None:
    global _compile_worker_current_device
    _compile_worker_current_device = device