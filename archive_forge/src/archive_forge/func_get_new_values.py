from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
def get_new_values(self, values, fill_value=None):
    if values.ndim == 1:
        values = values[:, np.newaxis]
    sorted_values = self._make_sorted_values(values)
    length, width = self.full_shape
    stride = values.shape[1]
    result_width = width * stride
    result_shape = (length, result_width)
    mask = self.mask
    mask_all = self.mask_all
    if mask_all and len(values):
        new_values = sorted_values.reshape(length, width, stride).swapaxes(1, 2).reshape(result_shape)
        new_mask = np.ones(result_shape, dtype=bool)
        return (new_values, new_mask)
    dtype = values.dtype
    if mask_all:
        dtype = values.dtype
        new_values = np.empty(result_shape, dtype=dtype)
    elif isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        new_values = cls._empty(result_shape, dtype=dtype)
        new_values[:] = fill_value
    else:
        dtype, fill_value = maybe_promote(dtype, fill_value)
        new_values = np.empty(result_shape, dtype=dtype)
        new_values.fill(fill_value)
    name = dtype.name
    new_mask = np.zeros(result_shape, dtype=bool)
    if needs_i8_conversion(values.dtype):
        sorted_values = sorted_values.view('i8')
        new_values = new_values.view('i8')
    else:
        sorted_values = sorted_values.astype(name, copy=False)
    libreshape.unstack(sorted_values, mask.view('u1'), stride, length, width, new_values, new_mask.view('u1'))
    if needs_i8_conversion(values.dtype):
        new_values = new_values.view('M8[ns]')
        new_values = ensure_wrapped_if_datetimelike(new_values)
        new_values = new_values.view(values.dtype)
    return (new_values, new_mask)