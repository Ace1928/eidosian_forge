import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def pandas_cat_null(data: DataFrame) -> DataFrame:
    """Handle categorical dtype and nullable extension types from pandas."""
    import pandas as pd
    cat_columns = []
    nul_columns = []
    for col, dtype in zip(data.columns, data.dtypes):
        if is_pd_cat_dtype(dtype):
            cat_columns.append(col)
        elif is_pa_ext_categorical_dtype(dtype):
            raise ValueError('pyarrow dictionary type is not supported. Use pandas category instead.')
        elif is_nullable_dtype(dtype):
            nul_columns.append(col)
    if cat_columns or nul_columns:
        transformed = data.copy(deep=False)
    else:
        transformed = data

    def cat_codes(ser: pd.Series) -> pd.Series:
        if is_pd_cat_dtype(ser.dtype):
            return ser.cat.codes
        assert is_pa_ext_categorical_dtype(ser.dtype)
        return ser.array.__arrow_array__().combine_chunks().dictionary_encode().indices
    if cat_columns:
        transformed[cat_columns] = transformed[cat_columns].apply(cat_codes).astype(np.float32).replace(-1.0, np.NaN)
    if nul_columns:
        transformed[nul_columns] = transformed[nul_columns].astype(np.float32)
    return transformed