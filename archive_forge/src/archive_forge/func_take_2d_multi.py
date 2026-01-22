from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike
def take_2d_multi(arr: np.ndarray, indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]], fill_value=np.nan) -> np.ndarray:
    """
    Specialized Cython take which sets NaN values in one pass.
    """
    assert indexer is not None
    assert indexer[0] is not None
    assert indexer[1] is not None
    row_idx, col_idx = indexer
    row_idx = ensure_platform_int(row_idx)
    col_idx = ensure_platform_int(col_idx)
    indexer = (row_idx, col_idx)
    mask_info = None
    dtype, fill_value = maybe_promote(arr.dtype, fill_value)
    if dtype != arr.dtype:
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()
        mask_info = ((row_mask, col_mask), (row_needs, col_needs))
        if not (row_needs or col_needs):
            dtype, fill_value = (arr.dtype, arr.dtype.type())
    out_shape = (len(row_idx), len(col_idx))
    out = np.empty(out_shape, dtype=dtype)
    func = _take_2d_multi_dict.get((arr.dtype.name, out.dtype.name), None)
    if func is None and arr.dtype != out.dtype:
        func = _take_2d_multi_dict.get((out.dtype.name, out.dtype.name), None)
        if func is not None:
            func = _convert_wrapper(func, out.dtype)
    if func is not None:
        func(arr, indexer, out=out, fill_value=fill_value)
    else:
        _take_2d_multi_object(arr, indexer, out, fill_value=fill_value, mask_info=mask_info)
    return out