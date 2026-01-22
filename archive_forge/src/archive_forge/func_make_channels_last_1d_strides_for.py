from __future__ import annotations
import operator
import warnings
import weakref
from contextlib import nullcontext
from enum import Enum
from functools import cmp_to_key, reduce
from typing import (
import torch
from torch import sym_float, sym_int, sym_max
def make_channels_last_1d_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    torch._check(len(shape) == 3, lambda: 'Only tensors of rank 3 can use the channels_last_1d memory format')
    multiplier = 1
    strides = [0] * 3
    for idx in (1, -1, 0):
        strides[idx] = multiplier
        multiplier *= shape[idx]
    return tuple(strides)