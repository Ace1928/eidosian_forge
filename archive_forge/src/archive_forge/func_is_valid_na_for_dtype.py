from __future__ import annotations
from decimal import Decimal
from functools import partial
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import lib
import pandas._libs.missing as libmissing
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import is_list_like
def is_valid_na_for_dtype(obj, dtype: DtypeObj) -> bool:
    """
    isna check that excludes incompatible dtypes

    Parameters
    ----------
    obj : object
    dtype : np.datetime64, np.timedelta64, DatetimeTZDtype, or PeriodDtype

    Returns
    -------
    bool
    """
    if not lib.is_scalar(obj) or not isna(obj):
        return False
    elif dtype.kind == 'M':
        if isinstance(dtype, np.dtype):
            return not isinstance(obj, (np.timedelta64, Decimal))
        return not isinstance(obj, (np.timedelta64, np.datetime64, Decimal))
    elif dtype.kind == 'm':
        return not isinstance(obj, (np.datetime64, Decimal))
    elif dtype.kind in 'iufc':
        return obj is not NaT and (not isinstance(obj, (np.datetime64, np.timedelta64)))
    elif dtype.kind == 'b':
        return lib.is_float(obj) or obj is None or obj is libmissing.NA
    elif dtype == _dtype_str:
        return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal, float))
    elif dtype == _dtype_object:
        return True
    elif isinstance(dtype, PeriodDtype):
        return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal))
    elif isinstance(dtype, IntervalDtype):
        return lib.is_float(obj) or obj is None or obj is libmissing.NA
    elif isinstance(dtype, CategoricalDtype):
        return is_valid_na_for_dtype(obj, dtype.categories.dtype)
    return not isinstance(obj, (np.datetime64, np.timedelta64, Decimal))