from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
def is_interval_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Interval dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.IntervalDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Interval dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_interval_dtype
    >>> is_interval_dtype(object)
    False
    >>> is_interval_dtype(pd.IntervalDtype())
    True
    >>> is_interval_dtype([1, 2, 3])
    False
    >>>
    >>> interval = pd.Interval(1, 2, closed="right")
    >>> is_interval_dtype(interval)
    False
    >>> is_interval_dtype(pd.IntervalIndex([interval]))
    True
    """
    warnings.warn('is_interval_dtype is deprecated and will be removed in a future version. Use `isinstance(dtype, pd.IntervalDtype)` instead', DeprecationWarning, stacklevel=2)
    if isinstance(arr_or_dtype, ExtensionDtype):
        return arr_or_dtype.type is Interval
    if arr_or_dtype is None:
        return False
    return IntervalDtype.is_dtype(arr_or_dtype)