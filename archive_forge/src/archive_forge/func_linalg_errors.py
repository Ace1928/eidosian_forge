from __future__ import annotations
import functools
import math
from typing import Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, KeepDims, normalizer
def linalg_errors(func):

    @functools.wraps(func)
    def wrapped(*args, **kwds):
        try:
            return func(*args, **kwds)
        except torch._C._LinAlgError as e:
            raise LinAlgError(*e.args)
    return wrapped