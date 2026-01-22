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
def histogram_scatter_add_2d(histogram: Tensor, index1: Tensor, index2: Tensor, source: Tensor):
    assert len(histogram.shape) == 2
    assert histogram.dtype == torch.float32
    assert source.dtype == torch.float32
    assert index1.dtype == torch.int32
    assert index2.dtype == torch.int32
    assert histogram.device.type == 'cuda'
    assert index1.device.type == 'cuda'
    assert index2.device.type == 'cuda'
    assert source.device.type == 'cuda'
    maxdim1 = ct.c_int32(histogram.shape[0])
    n = ct.c_int32(index1.numel())
    is_on_gpu([histogram, index1, index2, source])
    lib.chistogram_scatter_add_2d(get_ptr(histogram), get_ptr(index1), get_ptr(index2), get_ptr(source), maxdim1, n)