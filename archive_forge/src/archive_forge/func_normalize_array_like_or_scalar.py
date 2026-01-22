from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_array_like_or_scalar(x, parm=None):
    if _dtypes_impl.is_scalar_or_symbolic(x):
        return x
    return normalize_array_like(x, parm)