from __future__ import annotations
from collections.abc import Sequence
from typing import (
import warnings
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
import pandas.core.common as com
def sanitize_array(data, index: Index | None, dtype: DtypeObj | None=None, copy: bool=False, *, allow_2d: bool=False) -> ArrayLike:
    """
    Sanitize input data to an ndarray or ExtensionArray, copy if specified,
    coerce to the dtype if specified.

    Parameters
    ----------
    data : Any
    index : Index or None, default None
    dtype : np.dtype, ExtensionDtype, or None, default None
    copy : bool, default False
    allow_2d : bool, default False
        If False, raise if we have a 2D Arraylike.

    Returns
    -------
    np.ndarray or ExtensionArray
    """
    original_dtype = dtype
    if isinstance(data, ma.MaskedArray):
        data = sanitize_masked_array(data)
    if isinstance(dtype, NumpyEADtype):
        dtype = dtype.numpy_dtype
    object_index = False
    if isinstance(data, ABCIndex) and data.dtype == object and (dtype is None):
        object_index = True
    data = extract_array(data, extract_numpy=True, extract_range=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        if dtype is None:
            dtype = data.dtype
        data = lib.item_from_zerodim(data)
    elif isinstance(data, range):
        data = range_to_ndarray(data)
        copy = False
    if not is_list_like(data):
        if index is None:
            raise ValueError('index must be specified when data is not list-like')
        if isinstance(data, str) and using_pyarrow_string_dtype() and (original_dtype is None):
            from pandas.core.arrays.string_ import StringDtype
            dtype = StringDtype('pyarrow_numpy')
        data = construct_1d_arraylike_from_scalar(data, len(index), dtype)
        return data
    elif isinstance(data, ABCExtensionArray):
        if dtype is not None:
            subarr = data.astype(dtype, copy=copy)
        elif copy:
            subarr = data.copy()
        else:
            subarr = data
    elif isinstance(dtype, ExtensionDtype):
        _sanitize_non_ordered(data)
        cls = dtype.construct_array_type()
        subarr = cls._from_sequence(data, dtype=dtype, copy=copy)
    elif isinstance(data, np.ndarray):
        if isinstance(data, np.matrix):
            data = data.A
        if dtype is None:
            subarr = data
            if data.dtype == object:
                subarr = maybe_infer_to_datetimelike(data)
                if object_index and using_pyarrow_string_dtype() and is_string_dtype(subarr):
                    subarr = data
            elif data.dtype.kind == 'U' and using_pyarrow_string_dtype():
                from pandas.core.arrays.string_ import StringDtype
                dtype = StringDtype(storage='pyarrow_numpy')
                subarr = dtype.construct_array_type()._from_sequence(data, dtype=dtype)
            if subarr is data and copy:
                subarr = subarr.copy()
        else:
            subarr = _try_cast(data, dtype, copy)
    elif hasattr(data, '__array__'):
        data = np.array(data, copy=copy)
        return sanitize_array(data, index=index, dtype=dtype, copy=False, allow_2d=allow_2d)
    else:
        _sanitize_non_ordered(data)
        data = list(data)
        if len(data) == 0 and dtype is None:
            subarr = np.array([], dtype=np.float64)
        elif dtype is not None:
            subarr = _try_cast(data, dtype, copy)
        else:
            subarr = maybe_convert_platform(data)
            if subarr.dtype == object:
                subarr = cast(np.ndarray, subarr)
                subarr = maybe_infer_to_datetimelike(subarr)
    subarr = _sanitize_ndim(subarr, data, dtype, index, allow_2d=allow_2d)
    if isinstance(subarr, np.ndarray):
        dtype = cast(np.dtype, dtype)
        subarr = _sanitize_str_dtypes(subarr, data, dtype, copy)
    return subarr