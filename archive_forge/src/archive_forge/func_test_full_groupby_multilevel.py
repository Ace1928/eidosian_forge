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
@pytest.mark.parametrize('grouper', [lambda df: ['a'], lambda df: ['a', 'b'], lambda df: df['a'], lambda df: [df['a'], df['b']], pytest.param(lambda df: [df['a'] > 2, df['b'] > 1], marks=pytest.mark.xfail(not DASK_EXPR_ENABLED, reason='not yet supported'))])
@pytest.mark.parametrize('reverse', [True, False])
def test_full_groupby_multilevel(grouper, reverse):
    index = [0, 1, 3, 5, 6, 8, 9, 9, 9]
    if reverse:
        index = index[::-1]
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'd': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=index)
    ddf = dd.from_pandas(df, npartitions=3)

    def func(df):
        return df.assign(b=df.d - df.d.mean())
    with pytest.warns(UserWarning, match='`meta` is not specified'):
        assert_eq(df.groupby(grouper(df)).apply(func, **INCLUDE_GROUPS), ddf.groupby(grouper(ddf)).apply(func, **INCLUDE_GROUPS))