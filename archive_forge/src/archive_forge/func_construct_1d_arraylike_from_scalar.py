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
def construct_1d_arraylike_from_scalar(value: Scalar, length: int, dtype: DtypeObj | None) -> ArrayLike:
    """
    create a np.ndarray / pandas type of specified shape and dtype
    filled with values

    Parameters
    ----------
    value : scalar value
    length : int
    dtype : pandas_dtype or np.dtype

    Returns
    -------
    np.ndarray / pandas type of length, filled with value

    """
    if dtype is None:
        try:
            dtype, value = infer_dtype_from_scalar(value)
        except OutOfBoundsDatetime:
            dtype = _dtype_obj
    if isinstance(dtype, ExtensionDtype):
        cls = dtype.construct_array_type()
        seq = [] if length == 0 else [value]
        subarr = cls._from_sequence(seq, dtype=dtype).repeat(length)
    else:
        if length and dtype.kind in 'iu' and isna(value):
            dtype = np.dtype('float64')
        elif lib.is_np_dtype(dtype, 'US'):
            dtype = np.dtype('object')
            if not isna(value):
                value = ensure_str(value)
        elif dtype.kind in 'mM':
            value = _maybe_box_and_unbox_datetimelike(value, dtype)
        subarr = np.empty(length, dtype=dtype)
        if length:
            subarr.fill(value)
    return subarr