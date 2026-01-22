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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='Not public')
def test_reduction_method():
    df = pd.DataFrame({'x': range(50), 'y': range(50, 100)})
    ddf = dd.from_pandas(df, npartitions=4)
    chunk = lambda x, val=0: (x >= val).sum()
    agg = lambda x: x.sum()
    res = ddf.x.reduction(chunk, aggregate=agg)
    assert_eq(res, df.x.count())
    res = ddf.reduction(chunk, aggregate=agg)
    assert res._name == ddf.reduction(chunk, aggregate=agg)._name
    assert_eq(res, df.count())
    res2 = ddf.reduction(chunk, aggregate=agg, chunk_kwargs={'val': 25})
    assert res2._name == ddf.reduction(chunk, aggregate=agg, chunk_kwargs={'val': 25})._name
    assert res2._name != res._name
    assert_eq(res2, (df >= 25).sum())

    def sum_and_count(x):
        return pd.DataFrame({'sum': x.sum(), 'count': x.count()})
    res = ddf.reduction(sum_and_count, aggregate=lambda x: x.groupby(level=0).sum())
    assert_eq(res, pd.DataFrame({'sum': df.sum(), 'count': df.count()}))