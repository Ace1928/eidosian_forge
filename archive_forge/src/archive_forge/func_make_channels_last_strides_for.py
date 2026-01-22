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
def make_channels_last_strides_for(shape: ShapeType) -> Tuple[int, ...]:
    ndim = len(shape) if isinstance(shape, Sequence) else 1
    if ndim == 3:
        return make_channels_last_1d_strides_for(shape)
    elif ndim == 4:
        return make_channels_last_2d_strides_for(shape)
    elif ndim == 5:
        return make_channels_last_3d_strides_for(shape)
    else:
        raise RuntimeError(f'no channels last format strides exist in {ndim} dimensions')