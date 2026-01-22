import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def df_equals(df1, df2, check_dtypes=True):
    """Tests if df1 and df2 are equal.

    Args:
        df1: (pandas or modin DataFrame or series) dataframe to test if equal.
        df2: (pandas or modin DataFrame or series) dataframe to test if equal.

    Returns:
        True if df1 is equal to df2.
    """
    from modin.pandas.groupby import DataFrameGroupBy
    groupby_types = (pandas.core.groupby.DataFrameGroupBy, DataFrameGroupBy)
    if hasattr(df1, 'index') and hasattr(df2, 'index') and (len(df1) == 0) and (len(df2) == 0):
        if type(df1).__name__ == type(df2).__name__:
            if hasattr(df1, 'name') and hasattr(df2, 'name') and (df1.name == df2.name):
                return
            if hasattr(df1, 'columns') and hasattr(df2, 'columns') and df1.columns.equals(df2.columns):
                return
        assert False
    if isinstance(df1, (list, tuple)) and all((isinstance(d, (pd.DataFrame, pd.Series, pandas.DataFrame, pandas.Series)) for d in df1)):
        assert isinstance(df2, type(df1)), 'Different type of collection'
        assert len(df1) == len(df2), 'Different length result'
        return (df_equals(d1, d2) for d1, d2 in zip(df1, df2))
    if check_dtypes:
        assert_dtypes_equal(df1, df2)
    if isinstance(df1, (pd.DataFrame, pd.Series)):
        df1 = to_pandas(df1)
    if isinstance(df2, (pd.DataFrame, pd.Series)):
        df2 = to_pandas(df2)
    if isinstance(df1, pandas.DataFrame) and isinstance(df2, pandas.DataFrame):
        assert_empty_frame_equal(df1, df2)
    if isinstance(df1, pandas.DataFrame) and isinstance(df2, pandas.DataFrame):
        assert_frame_equal(df1, df2, check_dtype=False, check_datetimelike_compat=True, check_index_type=False, check_column_type=False, check_categorical=False)
        df_categories_equals(df1, df2)
    elif isinstance(df1, pandas.Index) and isinstance(df2, pandas.Index):
        assert_index_equal(df1, df2)
    elif isinstance(df1, pandas.Series) and isinstance(df2, pandas.Series):
        assert_series_equal(df1, df2, check_dtype=False, check_series_type=False)
    elif hasattr(df1, 'dtype') and hasattr(df2, 'dtype') and isinstance(df1.dtype, pandas.core.dtypes.dtypes.ExtensionDtype) and isinstance(df2.dtype, pandas.core.dtypes.dtypes.ExtensionDtype):
        assert_extension_array_equal(df1, df2)
    elif isinstance(df1, groupby_types) and isinstance(df2, groupby_types):
        for g1, g2 in zip(df1, df2):
            assert g1[0] == g2[0]
            df_equals(g1[1], g2[1])
    elif isinstance(df1, pandas.Series) and isinstance(df2, pandas.Series) and df1.empty and df2.empty:
        assert all(df1.index == df2.index)
        assert df1.dtypes == df2.dtypes
    elif isinstance(df1, pandas.core.arrays.NumpyExtensionArray):
        assert isinstance(df2, pandas.core.arrays.NumpyExtensionArray)
        assert df1 == df2
    elif isinstance(df1, np.recarray) and isinstance(df2, np.recarray):
        np.testing.assert_array_equal(df1, df2)
    else:
        res = df1 != df2
        if res.any() if isinstance(res, np.ndarray) else res:
            np.testing.assert_almost_equal(df1, df2)