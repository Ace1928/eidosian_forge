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
@pytest.mark.parametrize('func', ['cumsum', 'cumprod'])
def test_groupby_numeric_only_false(func):
    df = pd.DataFrame({'int': [1, 2, 3, 4, 5, 6, 7, 8], 'float': [1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0], 'dt': [pd.NaT] + [datetime(2010, i, 1) for i in range(1, 8)], 'A': 1})
    ddf = dd.from_pandas(df, npartitions=2)
    if PANDAS_GE_200:
        ctx = pytest.raises(TypeError, match='does not support')
        with ctx:
            getattr(ddf.groupby('A'), func)(numeric_only=False)
        with ctx:
            getattr(df.groupby('A'), func)(numeric_only=False)
        with ctx:
            getattr(ddf.groupby('A'), func)()
        with ctx:
            getattr(df.groupby('A'), func)()
    else:
        ctx = pytest.warns(FutureWarning, match='Dropping invalid columns')
        with ctx:
            dd_result = getattr(ddf.groupby('A'), func)(numeric_only=False)
        with ctx:
            pd_result = getattr(df.groupby('A'), func)(numeric_only=False)
        assert_eq(dd_result, pd_result)
        if PANDAS_GE_150:
            ctx = pytest.warns(FutureWarning, match='default value of numeric_only')
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            dd_result = getattr(ddf.groupby('A'), func)()
        with ctx:
            pd_result = getattr(df.groupby('A'), func)()
        assert_eq(dd_result, pd_result)