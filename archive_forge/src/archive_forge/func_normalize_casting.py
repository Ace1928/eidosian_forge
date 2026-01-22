from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_casting(arg, parm=None):
    if arg not in ['no', 'equiv', 'safe', 'same_kind', 'unsafe']:
        raise ValueError(f"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe' (got '{arg}')")
    return arg