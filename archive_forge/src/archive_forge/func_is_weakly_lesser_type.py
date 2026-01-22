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
def is_weakly_lesser_type(a: type, b: type) -> bool:
    """
    Compares two types, a and b, returning True if a is weakly "less" than b.

    The comparison is determined by the following type ordering: bool, int, float, complex.
    """
    a, b = (_maybe_get_pytype(a), _maybe_get_pytype(b))
    if a not in _ordered_types or b not in _ordered_types:
        raise RuntimeError(f'Expected builtin numeric types, found {a}, {b}')
    for typ in _ordered_types:
        if a == typ:
            return True
        if b == typ:
            return False
    raise RuntimeError('Unexpected termination!')