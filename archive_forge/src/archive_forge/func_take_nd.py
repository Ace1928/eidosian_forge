from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike
def take_nd(arr: ArrayLike, indexer, axis: AxisInt=0, fill_value=lib.no_default, allow_fill: bool=True) -> ArrayLike:
    """
    Specialized Cython take which sets NaN values in one pass

    This dispatches to ``take`` defined on ExtensionArrays.

    Note: this function assumes that the indexer is a valid(ated) indexer with
    no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take, subarrays corresponding to -1 value
        indices are filed with fill_value
    axis : int, default 0
        Axis to take from
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask.  Result is
        undefined if allow_fill == False and -1 is present in indexer.

    Returns
    -------
    subarray : np.ndarray or ExtensionArray
        May be the same type as the input, or cast to an ndarray.
    """
    if fill_value is lib.no_default:
        fill_value = na_value_for_dtype(arr.dtype, compat=False)
    elif lib.is_np_dtype(arr.dtype, 'mM'):
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
    if not isinstance(arr, np.ndarray):
        if not is_1d_only_ea_dtype(arr.dtype):
            arr = cast('NDArrayBackedExtensionArray', arr)
            return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis)
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)
    arr = np.asarray(arr)
    return _take_nd_ndarray(arr, indexer, axis, fill_value, allow_fill)