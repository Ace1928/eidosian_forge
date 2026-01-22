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
def infer_dtype_from_array(arr) -> tuple[DtypeObj, ArrayLike]:
    """
    Infer the dtype from an array.

    Parameters
    ----------
    arr : array

    Returns
    -------
    tuple (pandas-compat dtype, array)


    Examples
    --------
    >>> np.asarray([1, '1'])
    array(['1', '1'], dtype='<U21')

    >>> infer_dtype_from_array([1, '1'])
    (dtype('O'), [1, '1'])
    """
    if isinstance(arr, np.ndarray):
        return (arr.dtype, arr)
    if not is_list_like(arr):
        raise TypeError("'arr' must be list-like")
    arr_dtype = getattr(arr, 'dtype', None)
    if isinstance(arr_dtype, ExtensionDtype):
        return (arr.dtype, arr)
    elif isinstance(arr, ABCSeries):
        return (arr.dtype, np.asarray(arr))
    inferred = lib.infer_dtype(arr, skipna=False)
    if inferred in ['string', 'bytes', 'mixed', 'mixed-integer']:
        return (np.dtype(np.object_), arr)
    arr = np.asarray(arr)
    return (arr.dtype, arr)