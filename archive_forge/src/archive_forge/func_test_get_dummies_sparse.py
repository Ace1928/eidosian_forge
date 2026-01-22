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
@check_pandas_issue_45618_warning
def test_get_dummies_sparse():
    s = pd.Series(pd.Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c']))
    ds = dd.from_pandas(s, 2)
    exp = pd.get_dummies(s, sparse=True)
    res = dd.get_dummies(ds, sparse=True)
    with ignore_numpy_bool8_deprecation():
        assert_eq(exp, res)
    dtype = res.compute().a.dtype
    assert dtype.fill_value == _get_dummies_dtype_default(0)
    assert dtype.subtype == _get_dummies_dtype_default
    assert isinstance(res.a.compute().dtype, pd.SparseDtype)
    exp = pd.get_dummies(s.to_frame(name='a'), sparse=True)
    res = dd.get_dummies(ds.to_frame(name='a'), sparse=True)
    with ignore_numpy_bool8_deprecation():
        assert_eq(exp, res)
    assert isinstance(res.a_a.compute().dtype, pd.SparseDtype)