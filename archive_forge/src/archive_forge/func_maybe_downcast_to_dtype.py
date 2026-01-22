from __future__ import annotations
import datetime as dt
import functools
from typing import (
import warnings
import numpy as np
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import (
from pandas._libs.missing import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.timedeltas import array_to_timedelta64
from pandas.compat.numpy import np_version_gt2
from pandas.errors import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
from pandas.core.dtypes.missing import (
from pandas.io._util import _arrow_dtype_mapping
def maybe_downcast_to_dtype(result: ArrayLike, dtype: str | np.dtype) -> ArrayLike:
    """
    try to cast to the specified dtype (e.g. convert back to bool/int
    or could be an astype of float64->float32
    """
    if isinstance(result, ABCSeries):
        result = result._values
    do_round = False
    if isinstance(dtype, str):
        if dtype == 'infer':
            inferred_type = lib.infer_dtype(result, skipna=False)
            if inferred_type == 'boolean':
                dtype = 'bool'
            elif inferred_type == 'integer':
                dtype = 'int64'
            elif inferred_type == 'datetime64':
                dtype = 'datetime64[ns]'
            elif inferred_type in ['timedelta', 'timedelta64']:
                dtype = 'timedelta64[ns]'
            elif inferred_type == 'floating':
                dtype = 'int64'
                if issubclass(result.dtype.type, np.number):
                    do_round = True
            else:
                dtype = 'object'
        dtype = np.dtype(dtype)
    if not isinstance(dtype, np.dtype):
        raise TypeError(dtype)
    converted = maybe_downcast_numeric(result, dtype, do_round)
    if converted is not result:
        return converted
    if dtype.kind in 'mM' and result.dtype.kind in 'if':
        result = result.astype(dtype)
    elif dtype.kind == 'm' and result.dtype == _dtype_obj:
        result = cast(np.ndarray, result)
        result = array_to_timedelta64(result)
    elif dtype == np.dtype('M8[ns]') and result.dtype == _dtype_obj:
        result = cast(np.ndarray, result)
        return np.asarray(maybe_cast_to_datetime(result, dtype=dtype))
    return result