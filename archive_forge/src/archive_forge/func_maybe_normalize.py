from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def maybe_normalize(arg, parm):
    """Normalize arg if a normalizer is registered."""
    normalizer = normalizers.get(parm.annotation, None)
    return normalizer(arg, parm) if normalizer else arg