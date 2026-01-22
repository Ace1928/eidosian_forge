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
def maybe_downcast_numeric(result: ArrayLike, dtype: DtypeObj, do_round: bool=False) -> ArrayLike:
    """
    Subset of maybe_downcast_to_dtype restricted to numeric dtypes.

    Parameters
    ----------
    result : ndarray or ExtensionArray
    dtype : np.dtype or ExtensionDtype
    do_round : bool

    Returns
    -------
    ndarray or ExtensionArray
    """
    if not isinstance(dtype, np.dtype) or not isinstance(result.dtype, np.dtype):
        return result

    def trans(x):
        if do_round:
            return x.round()
        return x
    if dtype.kind == result.dtype.kind:
        if result.dtype.itemsize <= dtype.itemsize and result.size:
            return result
    if dtype.kind in 'biu':
        if not result.size:
            return trans(result).astype(dtype)
        if isinstance(result, np.ndarray):
            element = result.item(0)
        else:
            element = result.iloc[0]
        if not isinstance(element, (np.integer, np.floating, int, float, bool)):
            return result
        if issubclass(result.dtype.type, (np.object_, np.number)) and notna(result).all():
            new_result = trans(result).astype(dtype)
            if new_result.dtype.kind == 'O' or result.dtype.kind == 'O':
                if (new_result == result).all():
                    return new_result
            elif np.allclose(new_result, result, rtol=0):
                return new_result
    elif issubclass(dtype.type, np.floating) and result.dtype.kind != 'b' and (not is_string_dtype(result.dtype)):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'overflow encountered in cast', RuntimeWarning)
            new_result = result.astype(dtype)
        size_tols = {4: 0.0005, 8: 5e-08, 16: 5e-16}
        atol = size_tols.get(new_result.dtype.itemsize, 0.0)
        if np.allclose(new_result, result, equal_nan=True, rtol=0.0, atol=atol):
            return new_result
    elif dtype.kind == result.dtype.kind == 'c':
        new_result = result.astype(dtype)
        if np.array_equal(new_result, result, equal_nan=True):
            return new_result
    return result