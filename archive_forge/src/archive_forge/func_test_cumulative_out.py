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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='out ignored in dask-expr')
@pytest.mark.parametrize('cls', ['DataFrame', 'Series'])
def test_cumulative_out(cls):
    index = [f'row{i:03d}' for i in range(100)]
    df = pd.DataFrame(np.random.randn(100, 5), columns=list('abcde'), index=index)
    ddf = dd.from_pandas(df, 5)
    ddf_out = dd.from_pandas(pd.DataFrame([], columns=list('abcde'), index=index), 1)
    if cls == 'Series':
        df = df['a']
        ddf = ddf['a']
        ddf_out = ddf_out['a']
    ctx = pytest.warns(FutureWarning, match="the 'out' keyword is deprecated")
    with ctx:
        ddf.cumsum(out=ddf_out)
    assert_eq(ddf_out, df.cumsum())
    with ctx:
        ddf.cumprod(out=ddf_out)
    assert_eq(ddf_out, df.cumprod())
    with ctx:
        ddf.cummin(out=ddf_out)
    assert_eq(ddf_out, df.cummin())
    with ctx:
        ddf.cummax(out=ddf_out)
    assert_eq(ddf_out, df.cummax())
    with ctx:
        np.cumsum(ddf, out=ddf_out)
    assert_eq(ddf_out, df.cumsum())
    with ctx:
        np.cumprod(ddf, out=ddf_out)
    assert_eq(ddf_out, df.cumprod())