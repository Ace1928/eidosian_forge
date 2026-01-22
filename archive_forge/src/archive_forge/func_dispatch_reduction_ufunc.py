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
def dispatch_reduction_ufunc(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
    """
    Dispatch ufunc reductions to self's reduction methods.
    """
    assert method == 'reduce'
    if len(inputs) != 1 or inputs[0] is not self:
        return NotImplemented
    if ufunc.__name__ not in REDUCTION_ALIASES:
        return NotImplemented
    method_name = REDUCTION_ALIASES[ufunc.__name__]
    if not hasattr(self, method_name):
        return NotImplemented
    if self.ndim > 1:
        if isinstance(self, ABCNDFrame):
            kwargs['numeric_only'] = False
        if 'axis' not in kwargs:
            kwargs['axis'] = 0
    return getattr(self, method_name)(skipna=False, **kwargs)