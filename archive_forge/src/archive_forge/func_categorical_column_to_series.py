from __future__ import annotations
import ctypes
import re
from typing import Any
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SettingWithCopyError
import pandas as pd
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
    """
    Convert a column holding categorical data to a pandas Series.

    Parameters
    ----------
    col : Column

    Returns
    -------
    tuple
        Tuple of pd.Series holding the data and the memory owner object
        that keeps the memory alive.
    """
    categorical = col.describe_categorical
    if not categorical['is_dictionary']:
        raise NotImplementedError('Non-dictionary categoricals not supported yet')
    cat_column = categorical['categories']
    if hasattr(cat_column, '_col'):
        categories = np.array(cat_column._col)
    else:
        raise NotImplementedError("Interchanging categorical columns isn't supported yet, and our fallback of using the `col._col` attribute (a ndarray) failed.")
    buffers = col.get_buffers()
    codes_buff, codes_dtype = buffers['data']
    codes = buffer_to_ndarray(codes_buff, codes_dtype, offset=col.offset, length=col.size())
    if len(categories) > 0:
        values = categories[codes % len(categories)]
    else:
        values = codes
    cat = pd.Categorical(values, categories=categories, ordered=categorical['is_ordered'])
    data = pd.Series(cat)
    data = set_nulls(data, col, buffers['validity'])
    return (data, buffers)