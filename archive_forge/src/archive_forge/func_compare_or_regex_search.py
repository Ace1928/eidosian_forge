from __future__ import annotations
import operator
import re
from re import Pattern
from typing import (
import numpy as np
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import isna
def compare_or_regex_search(a: ArrayLike, b: Scalar | Pattern, regex: bool, mask: npt.NDArray[np.bool_]) -> ArrayLike:
    """
    Compare two array-like inputs of the same shape or two scalar values

    Calls operator.eq or re.search, depending on regex argument. If regex is
    True, perform an element-wise regex matching.

    Parameters
    ----------
    a : array-like
    b : scalar or regex pattern
    regex : bool
    mask : np.ndarray[bool]

    Returns
    -------
    mask : array-like of bool
    """
    if isna(b):
        return ~mask

    def _check_comparison_types(result: ArrayLike | bool, a: ArrayLike, b: Scalar | Pattern):
        """
        Raises an error if the two arrays (a,b) cannot be compared.
        Otherwise, returns the comparison result as expected.
        """
        if is_bool(result) and isinstance(a, np.ndarray):
            type_names = [type(a).__name__, type(b).__name__]
            type_names[0] = f'ndarray(dtype={a.dtype})'
            raise TypeError(f'Cannot compare types {repr(type_names[0])} and {repr(type_names[1])}')
    if not regex or not should_use_regex(regex, b):
        op = lambda x: operator.eq(x, b)
    else:
        op = np.vectorize(lambda x: bool(re.search(b, x)) if isinstance(x, str) and isinstance(b, (str, Pattern)) else False)
    if isinstance(a, np.ndarray):
        a = a[mask]
    result = op(a)
    if isinstance(result, np.ndarray) and mask is not None:
        tmp = np.zeros(mask.shape, dtype=np.bool_)
        np.place(tmp, mask, result)
        result = tmp
    _check_comparison_types(result, a, b)
    return result