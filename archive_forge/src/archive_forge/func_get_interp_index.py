from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def get_interp_index(method, index: Index) -> Index:
    if method == 'linear':
        from pandas import Index
        index = Index(np.arange(len(index)))
    else:
        methods = {'index', 'values', 'nearest', 'time'}
        is_numeric_or_datetime = is_numeric_dtype(index.dtype) or isinstance(index.dtype, DatetimeTZDtype) or lib.is_np_dtype(index.dtype, 'mM')
        if method not in methods and (not is_numeric_or_datetime):
            raise ValueError(f'Index column must be numeric or datetime type when using {method} method other than linear. Try setting a numeric or datetime index column before interpolating.')
    if isna(index).any():
        raise NotImplementedError('Interpolation with NaNs in the index has not been implemented. Try filling those NaNs before interpolating.')
    return index