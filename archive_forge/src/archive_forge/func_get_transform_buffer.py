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
def get_transform_buffer(shape, dtype, device, to_order, from_order='row', transpose=False):
    init_func = torch.zeros
    dims = len(shape)
    if dims == 2:
        rows = shape[0]
    elif dims == 3:
        rows = shape[0] * shape[1]
    cols = shape[-1]
    state = (shape, to_order)
    if transpose:
        tmp = rows
        rows = cols
        cols = tmp
        state = (shape[::-1], to_order)
    if to_order == 'row' or to_order == 'col':
        return (init_func(shape, dtype=dtype, device=device), state)
    elif to_order == 'col32':
        cols = 32 * ((cols + 31) // 32)
        return (init_func((rows, cols), dtype=dtype, device=device), state)
    elif to_order == 'col_turing':
        cols = 32 * ((cols + 31) // 32)
        rows = 8 * ((rows + 7) // 8)
        return (init_func((rows, cols), dtype=dtype, device=device), state)
    elif to_order == 'col_ampere':
        cols = 32 * ((cols + 31) // 32)
        rows = 32 * ((rows + 31) // 32)
        return (init_func((rows, cols), dtype=dtype, device=device), state)
    else:
        raise NotImplementedError(f'To_order not supported: {to_order}')