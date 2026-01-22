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
def reduction_dims(shape: ShapeType, dims: Optional[Sequence]) -> Tuple[int, ...]:
    if dims is None:
        return tuple(range(len(shape)))
    dims = tuple((canonicalize_dim(len(shape), idx) for idx in dims))
    validate_no_repeating_dims(dims)
    return dims