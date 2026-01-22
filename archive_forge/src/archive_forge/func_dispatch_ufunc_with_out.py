from __future__ import annotations
import operator
from typing import Any
import numpy as np
from pandas._libs import lib
from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op
from pandas.core.dtypes.generic import ABCNDFrame
from pandas.core import roperator
from pandas.core.construction import extract_array
from pandas.core.ops.common import unpack_zerodim_and_defer
def dispatch_ufunc_with_out(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    If we have an `out` keyword, then call the ufunc without `out` and then
    set the result into the given `out`.
    """
    out = kwargs.pop('out')
    where = kwargs.pop('where', None)
    result = getattr(ufunc, method)(*inputs, **kwargs)
    if result is NotImplemented:
        return NotImplemented
    if isinstance(result, tuple):
        if not isinstance(out, tuple) or len(out) != len(result):
            raise NotImplementedError
        for arr, res in zip(out, result):
            _assign_where(arr, res, where)
        return out
    if isinstance(out, tuple):
        if len(out) == 1:
            out = out[0]
        else:
            raise NotImplementedError
    _assign_where(out, result, where)
    return out