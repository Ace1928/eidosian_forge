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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not supported')
@pytest.mark.parametrize('convert_dtype', [None, True, False])
def test_apply_convert_dtype(convert_dtype):
    """Make sure that explicit convert_dtype raises a warning with pandas>=2.1"""
    df = pd.DataFrame({'x': [2, 3, 4, 5], 'y': [10, 20, 30, 40]})
    ddf = dd.from_pandas(df, npartitions=2)
    kwargs = {} if convert_dtype is None else {'convert_dtype': convert_dtype}
    pd_should_warn = PANDAS_GE_210 and convert_dtype is not None
    meta_val = ddf.x._meta_nonempty.iloc[0]

    def func(x):
        assert x != meta_val
        return x + 1
    with _check_warning(pd_should_warn, FutureWarning, 'the convert_dtype parameter'):
        expected = df.x.apply(func, **kwargs)
    with _check_warning(pd_should_warn, FutureWarning, 'the convert_dtype parameter'):
        result = ddf.x.apply(func, **kwargs, meta=expected)
    dask_should_warn = pytest.warns(FutureWarning, match="the 'convert_dtype' keyword")
    if convert_dtype is None:
        dask_should_warn = contextlib.nullcontext()
    with dask_should_warn:
        assert_eq(result, expected)