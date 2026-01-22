from __future__ import annotations
import inspect
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.errors import IntCastingNaNError
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
def astype_is_view(dtype: DtypeObj, new_dtype: DtypeObj) -> bool:
    """Checks if astype avoided copying the data.

    Parameters
    ----------
    dtype : Original dtype
    new_dtype : target dtype

    Returns
    -------
    True if new data is a view or not guaranteed to be a copy, False otherwise
    """
    if isinstance(dtype, np.dtype) and (not isinstance(new_dtype, np.dtype)):
        new_dtype, dtype = (dtype, new_dtype)
    if dtype == new_dtype:
        return True
    elif isinstance(dtype, np.dtype) and isinstance(new_dtype, np.dtype):
        return False
    elif is_string_dtype(dtype) and is_string_dtype(new_dtype):
        return True
    elif is_object_dtype(dtype) and new_dtype.kind == 'O':
        return True
    elif dtype.kind in 'mM' and new_dtype.kind in 'mM':
        dtype = getattr(dtype, 'numpy_dtype', dtype)
        new_dtype = getattr(new_dtype, 'numpy_dtype', new_dtype)
        return getattr(dtype, 'unit', None) == getattr(new_dtype, 'unit', None)
    numpy_dtype = getattr(dtype, 'numpy_dtype', None)
    new_numpy_dtype = getattr(new_dtype, 'numpy_dtype', None)
    if numpy_dtype is None and isinstance(dtype, np.dtype):
        numpy_dtype = dtype
    if new_numpy_dtype is None and isinstance(new_dtype, np.dtype):
        new_numpy_dtype = new_dtype
    if numpy_dtype is not None and new_numpy_dtype is not None:
        return numpy_dtype == new_numpy_dtype
    return True