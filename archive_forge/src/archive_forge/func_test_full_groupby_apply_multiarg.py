from __future__ import annotations
import contextlib
import operator
import warnings
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.dataframe import _compat
from dask.dataframe._compat import (
from dask.dataframe._pyarrow import to_pyarrow_string
from dask.dataframe.backends import grouper_dispatch
from dask.dataframe.groupby import NUMERIC_ONLY_NOT_IMPLEMENTED
from dask.dataframe.utils import assert_dask_graph, assert_eq, pyarrow_strings_enabled
from dask.utils import M
from dask.utils_test import _check_warning, hlg_layer
@pytest.mark.xfail(DASK_EXPR_ENABLED, reason="can't support collections in kwargs for apply")
def test_full_groupby_apply_multiarg():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
    ddf = dd.from_pandas(df, npartitions=3)

    def func(df, c, d=3):
        return df.assign(b=df.b - df.b.mean() + c * d)
    c = df.a.sum()
    d = df.b.mean()
    c_scalar = ddf.a.sum()
    d_scalar = ddf.b.mean()
    c_delayed = dask.delayed(lambda: c)()
    d_delayed = dask.delayed(lambda: d)()
    with pytest.warns(UserWarning, match='`meta` is not specified'):
        assert_eq(df.groupby('a').apply(func, c, d=d, **INCLUDE_GROUPS), ddf.groupby('a').apply(func, c, d=d_scalar, **INCLUDE_GROUPS))
        assert_eq(df.groupby('a').apply(func, c, **INCLUDE_GROUPS), ddf.groupby('a').apply(func, c, **INCLUDE_GROUPS))
        assert_eq(df.groupby('a').apply(func, c, d=d, **INCLUDE_GROUPS), ddf.groupby('a').apply(func, c, d=d, **INCLUDE_GROUPS))
        assert_eq(df.groupby('a').apply(func, c, **INCLUDE_GROUPS), ddf.groupby('a').apply(func, c_scalar, **INCLUDE_GROUPS), check_dtype=False)
    meta = df.groupby('a').apply(func, c, **INCLUDE_GROUPS)
    assert_eq(df.groupby('a').apply(func, c, **INCLUDE_GROUPS), ddf.groupby('a').apply(func, c_scalar, meta=meta, **INCLUDE_GROUPS))
    assert_eq(df.groupby('a').apply(func, c, d=d, **INCLUDE_GROUPS), ddf.groupby('a').apply(func, c, d=d_scalar, meta=meta, **INCLUDE_GROUPS))
    with pytest.raises(ValueError) as exc:
        ddf.groupby('a').apply(func, c, d=d_delayed, **INCLUDE_GROUPS)
    assert 'dask.delayed' in str(exc.value) and 'meta' in str(exc.value)
    with pytest.raises(ValueError) as exc:
        ddf.groupby('a').apply(func, c_delayed, d=d, **INCLUDE_GROUPS)
    assert 'dask.delayed' in str(exc.value) and 'meta' in str(exc.value)
    assert_eq(df.groupby('a').apply(func, c, **INCLUDE_GROUPS), ddf.groupby('a').apply(func, c_delayed, meta=meta, **INCLUDE_GROUPS))
    assert_eq(df.groupby('a').apply(func, c, d=d, **INCLUDE_GROUPS), ddf.groupby('a').apply(func, c, d=d_delayed, meta=meta, **INCLUDE_GROUPS))