from __future__ import annotations
from collections.abc import (
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas._libs.internals import (
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
from pandas.core.internals.base import (
from pandas.core.internals.blocks import (
from pandas.core.internals.ops import (
def make_na_array(dtype: DtypeObj, shape: Shape, fill_value) -> ArrayLike:
    if isinstance(dtype, DatetimeTZDtype):
        ts = Timestamp(fill_value).as_unit(dtype.unit)
        i8values = np.full(shape, ts._value)
        dt64values = i8values.view(f'M8[{dtype.unit}]')
        return DatetimeArray._simple_new(dt64values, dtype=dtype)
    elif is_1d_only_ea_dtype(dtype):
        dtype = cast(ExtensionDtype, dtype)
        cls = dtype.construct_array_type()
        missing_arr = cls._from_sequence([], dtype=dtype)
        ncols, nrows = shape
        assert ncols == 1, ncols
        empty_arr = -1 * np.ones((nrows,), dtype=np.intp)
        return missing_arr.take(empty_arr, allow_fill=True, fill_value=fill_value)
    elif isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        missing_arr = cls._empty(shape=shape, dtype=dtype)
        missing_arr[:] = fill_value
        return missing_arr
    else:
        missing_arr = np.empty(shape, dtype=dtype)
        missing_arr.fill(fill_value)
        if dtype.kind in 'mM':
            missing_arr = ensure_wrapped_if_datetimelike(missing_arr)
        return missing_arr