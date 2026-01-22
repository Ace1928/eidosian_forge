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
@pytest.mark.parametrize('split_out', [2, 3])
@pytest.mark.parametrize('column', [['b', 'c'], ['b', 'd'], ['b', 'e']])
def test_groupby_split_out_multiindex(split_out, column):
    df = pd.DataFrame({'a': np.arange(8), 'b': [1, 0, 0, 2, 1, 1, 2, 0], 'c': [0, 1] * 4, 'd': ['dog', 'cat', 'cat', 'dog', 'dog', 'dog', 'cat', 'bird']}).fillna(0)
    df['e'] = df['d'].astype('category')
    ddf = dd.from_pandas(df, npartitions=3)
    if column == ['b', 'e'] and PANDAS_GE_210 and (not PANDAS_GE_300):
        ctx = pytest.warns(FutureWarning, match='observed')
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        result_so1 = ddf.groupby(column, sort=False).a.mean(split_out=1).compute().dropna()
        result = ddf.groupby(column, sort=False).a.mean(split_out=split_out).compute().dropna()
    assert_eq(result, result_so1)