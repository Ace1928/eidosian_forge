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
@pytest.mark.parametrize('func', NUMERIC_ONLY_NOT_IMPLEMENTED)
@pytest.mark.parametrize('numeric_only', [False, None])
def test_groupby_numeric_only_not_implemented(func, numeric_only):
    """These should warn / error when numeric_only is set to its default / False"""
    df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 4, 3], 'C': ['a', 'b', 'c']})
    ddf = dd.from_pandas(df, npartitions=3)
    ctx = contextlib.nullcontext()
    ctx_warn = pytest.warns(FutureWarning, match='The default value of numeric_only')
    ctx_error = pytest.raises(NotImplementedError, match="'numeric_only=False' is not implemented in Dask")
    if numeric_only is None:
        if PANDAS_GE_150 and (not PANDAS_GE_200):
            ctx = ctx_warn
        elif PANDAS_GE_200:
            ctx = ctx_error
    else:
        ctx = ctx_error
    kwargs = {} if numeric_only is None else {'numeric_only': numeric_only}
    with ctx:
        getattr(ddf.groupby('A'), func)(**kwargs)