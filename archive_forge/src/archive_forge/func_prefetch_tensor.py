import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def prefetch_tensor(A, to_cpu=False):
    assert A.is_paged, 'Only paged tensors can be prefetched!'
    if to_cpu:
        deviceid = -1
    else:
        deviceid = A.page_deviceid
    num_bytes = dtype2bytes[A.dtype] * A.numel()
    lib.cprefetch(get_ptr(A), ct.c_size_t(num_bytes), ct.c_int32(deviceid))