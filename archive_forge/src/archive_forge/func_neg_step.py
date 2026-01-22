from __future__ import annotations
import builtins
import math
import operator
from typing import Sequence
import torch
from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import (
def neg_step(i, s):
    if not (isinstance(s, slice) and s.step is not None and (s.step < 0)):
        return s
    nonlocal tensor
    tensor = torch.flip(tensor, (i,))
    assert isinstance(s.start, int) or s.start is None
    assert isinstance(s.stop, int) or s.stop is None
    start = s.stop + 1 if s.stop else None
    stop = s.start + 1 if s.start else None
    return slice(start, stop, -s.step)