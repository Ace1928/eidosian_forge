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
@pytest.mark.parametrize('agg_func', ['cumprod', 'cumcount', 'cumsum', 'var', 'sum', 'mean', 'count', 'size', 'std', 'min', 'max', 'first', 'last', 'prod'])
def test_groupby_column_and_index_agg_funcs(agg_func):

    def call(g, m, **kwargs):
        return getattr(g, m)(**kwargs)
    df = pd.DataFrame({'idx': [1, 1, 1, 2, 2, 2], 'a': [1, 2, 1, 2, 1, 2], 'b': np.arange(6), 'c': [1, 1, 1, 2, 2, 2]}).set_index('idx')
    ddf = dd.from_pandas(df, npartitions=df.index.nunique())
    ddf_no_divs = dd.from_pandas(df, npartitions=df.index.nunique(), sort=False)
    expected = call(df.groupby(['idx', 'a']), agg_func)
    if agg_func in {'mean', 'var'}:
        expected = expected.astype(float)
    result = call(ddf.groupby(['idx', 'a']), agg_func)
    assert_eq(expected, result)
    result = call(ddf_no_divs.groupby(['idx', 'a']), agg_func)
    assert_eq(expected, result)
    aca_agg = {'sum', 'mean', 'var', 'size', 'std', 'count', 'first', 'last', 'prod'}
    if agg_func in aca_agg:
        result = ddf_no_divs.groupby(['idx', 'a']).agg(agg_func)
        assert_eq(expected, result)
    expected = call(df.groupby(['a', 'idx']), agg_func)
    if agg_func in {'mean', 'var'}:
        expected = expected.astype(float)
    result = call(ddf.groupby(['a', 'idx']), agg_func)
    assert_eq(expected, result)
    result = call(ddf_no_divs.groupby(['a', 'idx']), agg_func)
    assert_eq(expected, result)
    if agg_func in aca_agg:
        result = ddf_no_divs.groupby(['a', 'idx']).agg(agg_func)
        assert_eq(expected, result)
    expected = call(df.groupby('idx'), agg_func)
    if agg_func in {'mean', 'var'}:
        expected = expected.astype(float)
    result = call(ddf.groupby('idx'), agg_func)
    assert_eq(expected, result)
    result = call(ddf_no_divs.groupby('idx'), agg_func)
    assert_eq(expected, result)
    if agg_func in aca_agg:
        result = ddf_no_divs.groupby('idx').agg(agg_func)
        assert_eq(expected, result)