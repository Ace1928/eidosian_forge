from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
def looper_wrapper(values, start=None, end=None, labels=None, ngroups=None, min_periods: int=0, **kwargs):
    result_dtype = dtype_mapping[values.dtype]
    column_looper = make_looper(func, result_dtype, is_grouped_kernel, nopython, nogil, parallel)
    if is_grouped_kernel:
        result, na_positions = column_looper(values, labels, ngroups, min_periods, *kwargs.values())
    else:
        result, na_positions = column_looper(values, start, end, min_periods, *kwargs.values())
    if result.dtype.kind == 'i':
        for na_pos in na_positions.values():
            if len(na_pos) > 0:
                result = result.astype('float64')
                break
    for i, na_pos in na_positions.items():
        if len(na_pos) > 0:
            result[i, na_pos] = np.nan
    return result