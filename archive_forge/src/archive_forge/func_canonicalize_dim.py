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
def canonicalize_dim(rank: int, idx: int, wrap_scalar: bool=True) -> int:
    if rank < 0:
        msg = f'Rank cannot be negative but got {rank}'
        raise IndexError(msg)
    if rank == 0:
        if not wrap_scalar:
            msg = f'Dimension specified as {idx} but tensor has no dimensions'
            raise IndexError(msg)
        rank = 1
    if idx >= 0 and idx < rank:
        return idx
    if idx < 0:
        _idx = idx + rank
    else:
        _idx = idx
    if _idx < 0 or _idx >= rank:
        msg = f'Dimension out of range (expected to be in range of [{-rank}, {rank - 1}], but got {idx})'
        raise IndexError(msg)
    return _idx