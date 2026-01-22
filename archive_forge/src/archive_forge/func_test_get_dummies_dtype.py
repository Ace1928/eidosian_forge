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
def test_get_dummies_dtype():
    df = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c']), 'B': [0, 0, 1]})
    ddf = dd.from_pandas(df, 2)
    exp = pd.get_dummies(df, dtype='float64')
    res = dd.get_dummies(ddf, dtype='float64')
    assert_eq(exp, res)
    assert res.compute().A_a.dtype == 'float64'
    assert_eq(dd.get_dummies(df, dtype='float64'), exp)
    assert res.compute().A_a.dtype == 'float64'