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
def test_get_dummies_kwargs():
    s = pd.Series([1, 1, 1, 2, 2, 1, 3, 4], dtype='category')
    exp = pd.get_dummies(s, prefix='X', prefix_sep='-')
    ds = dd.from_pandas(s, 2)
    res = dd.get_dummies(ds, prefix='X', prefix_sep='-')
    assert_eq(res, exp)
    exp = pd.get_dummies(s, drop_first=True)
    res = dd.get_dummies(ds, drop_first=True)
    assert_eq(res, exp)
    s = pd.Series([1, 1, 1, 2, np.nan, 3, np.nan, 5], dtype='category')
    exp = pd.get_dummies(s)
    ds = dd.from_pandas(s, 2)
    res = dd.get_dummies(ds)
    assert_eq(res, exp)
    exp = pd.get_dummies(s, dummy_na=True)
    res = dd.get_dummies(ds, dummy_na=True)
    assert_eq(res, exp)