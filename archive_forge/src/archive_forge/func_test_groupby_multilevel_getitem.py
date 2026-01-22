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
@pytest.mark.parametrize('grouper', [lambda df: df.groupby('a')['b'], lambda df: df.groupby(['a', 'b']), lambda df: df.groupby(['a', 'b'])['c'], lambda df: df.groupby(df['a'])[['b', 'c']], lambda df: df.groupby('a')[['b', 'c']], lambda df: df.groupby('a')[['b']], lambda df: df.groupby(['a', 'b', 'c'])])
def test_groupby_multilevel_getitem(grouper, agg_func):
    if agg_func == 'nunique':
        return
    df = pd.DataFrame({'a': [1, 2, 3, 1, 2, 3], 'b': [1, 2, 1, 4, 2, 1], 'c': [1, 3, 2, 1, 1, 2], 'd': [1, 2, 1, 1, 2, 2]})
    ddf = dd.from_pandas(df, 2)
    dask_group = grouper(ddf)
    pandas_group = grouper(df)
    if isinstance(pandas_group, pd.core.groupby.SeriesGroupBy) and agg_func in ('cov', 'corr'):
        return
    dask_agg = getattr(dask_group, agg_func)
    pandas_agg = getattr(pandas_group, agg_func)
    if not DASK_EXPR_ENABLED:
        assert isinstance(dask_group, dd.groupby._GroupBy)
    assert isinstance(pandas_group, pd.core.groupby.GroupBy)
    if agg_func == 'mean':
        assert_eq(dask_agg(), pandas_agg().astype(float))
    else:
        a = dask_agg()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            b = pandas_agg()
        assert_eq(a, b)