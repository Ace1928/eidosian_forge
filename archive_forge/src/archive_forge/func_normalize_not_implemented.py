from __future__ import annotations
import functools
import inspect
import operator
import typing
import torch
from . import _dtypes, _dtypes_impl, _util
def normalize_not_implemented(arg, parm):
    if arg != parm.default:
        raise NotImplementedError(f"'{parm.name}' parameter is not supported.")