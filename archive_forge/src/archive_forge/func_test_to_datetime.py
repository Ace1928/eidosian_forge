from __future__ import annotations
import contextlib
import decimal
import warnings
import weakref
import xml.etree.ElementTree
from datetime import datetime, timedelta
from itertools import product
from operator import add
from textwrap import dedent
import numpy as np
import pandas as pd
import pytest
from pandas.errors import PerformanceWarning
from pandas.io.formats import format as pandas_format
import dask
import dask.array as da
import dask.dataframe as dd
import dask.dataframe.groupby
from dask import delayed
from dask.base import compute_as_if_collection
from dask.blockwise import fuse_roots
from dask.dataframe import _compat, methods
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.core import (
from dask.dataframe.utils import (
from dask.datasets import timeseries
from dask.utils import M, is_dataframe_like, is_series_like, put_lines
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.parametrize('gpu', [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_to_datetime(gpu):
    xd = pd if not gpu else pytest.importorskip('cudf')
    check_dtype = not gpu
    df = xd.DataFrame({'year': [2015, 2016], 'month': ['2', '3'], 'day': [4, 5]})
    df.index.name = 'ix'
    ddf = dd.from_pandas(df, npartitions=2)
    assert_eq(xd.to_datetime(df), dd.to_datetime(ddf), check_dtype=check_dtype)
    assert_eq(xd.to_datetime(df), dd.to_datetime(df), check_dtype=check_dtype)
    s = xd.Series(['3/11/2000', '3/12/2000', '3/13/2000'] * 100, index=['3/11/2000', '3/12/2000', '3/13/2000'] * 100)
    ds = dd.from_pandas(s, npartitions=10, sort=False)
    if not DASK_EXPR_ENABLED:
        if PANDAS_GE_200:
            ctx = pytest.warns(UserWarning, match="'infer_datetime_format' is deprecated")
        else:
            ctx = contextlib.nullcontext()
        ctx_expected = contextlib.nullcontext() if gpu else ctx
        with ctx_expected:
            expected = xd.to_datetime(s, infer_datetime_format=True)
        with ctx:
            result = dd.to_datetime(ds, infer_datetime_format=True)
        assert_eq(expected, result, check_dtype=check_dtype)
        with ctx:
            result = dd.to_datetime(s, infer_datetime_format=True)
        assert_eq(expected, result, check_dtype=check_dtype)
        with ctx_expected:
            expected = xd.to_datetime(s.index, infer_datetime_format=True)
        with ctx:
            result = dd.to_datetime(ds.index, infer_datetime_format=True)
        assert_eq(expected, result, check_divisions=False)
    if not gpu:
        assert_eq(xd.to_datetime(s, utc=True), dd.to_datetime(ds, utc=True))
        assert_eq(xd.to_datetime(s, utc=True), dd.to_datetime(s, utc=True))
    for arg in ('2021-08-03', 2021, s.index):
        with pytest.raises(NotImplementedError, match='non-index-able arguments'):
            dd.to_datetime(arg)