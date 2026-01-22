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
def infer_size_shapes(a: ShapeType, b: ShapeType) -> Tuple[int, ...]:
    ndim = max(len(a), len(b))
    expandedSizes = [0] * ndim
    for i in range(ndim - 1, -1, -1):
        offset = ndim - 1 - i
        dimA = len(a) - 1 - offset
        dimB = len(b) - 1 - offset
        sizeA = a[dimA] if dimA >= 0 else 1
        sizeB = b[dimB] if dimB >= 0 else 1
        torch._check(sizeA == sizeB or sizeA == 1 or sizeB == 1, lambda: f'The size of tensor a ({sizeA}) must match the size of tensor b ({sizeB}) at non-singleton dimension {i}')
        expandedSizes[i] = sizeB if sizeA == 1 else sizeA
    return tuple(expandedSizes)