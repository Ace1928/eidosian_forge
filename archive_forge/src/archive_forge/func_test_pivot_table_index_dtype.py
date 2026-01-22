from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_VERSION, tm
from dask.dataframe.reshape import _get_dummies_dtype_default
from dask.dataframe.utils import assert_eq
def test_pivot_table_index_dtype():
    df = pd.DataFrame({'A': pd.date_range(start='2019-08-01', periods=3, freq='1D'), 'B': pd.Categorical(list('abc')), 'C': [1, 2, 3]})
    ddf = dd.from_pandas(df, 2)
    res = dd.pivot_table(ddf, index='A', columns='B', values='C', aggfunc='count')
    assert res.index.dtype == np.dtype('datetime64[ns]')