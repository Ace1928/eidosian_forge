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
def same_shape(a: ShapeType, b: ShapeType, *, allow_rhs_unbacked=False) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if allow_rhs_unbacked:
            if isinstance(y, torch.SymInt):
                continue
        if x != y:
            return False
    return True