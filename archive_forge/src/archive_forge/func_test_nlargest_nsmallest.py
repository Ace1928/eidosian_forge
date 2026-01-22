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
def test_nlargest_nsmallest():
    from string import ascii_lowercase
    df = pd.DataFrame({'a': np.random.permutation(20), 'b': list(ascii_lowercase[:20]), 'c': np.random.permutation(20).astype('float64')})
    ddf = dd.from_pandas(df, npartitions=3)
    for m in ['nlargest', 'nsmallest']:
        f = lambda df=df, m=m, *args, **kwargs: getattr(df, m)(*args, **kwargs)
        res = f(ddf, m, 5, 'a')
        res2 = f(ddf, m, 5, 'a', split_every=2)
        sol = f(df, m, 5, 'a')
        assert_eq(res, sol)
        assert_eq(res2, sol)
        assert res._name != res2._name
        res = f(ddf, m, 5, ['a', 'c'])
        res2 = f(ddf, m, 5, ['a', 'c'], split_every=2)
        sol = f(df, m, 5, ['a', 'c'])
        assert_eq(res, sol)
        assert_eq(res2, sol)
        assert res._name != res2._name
        res = f(ddf.a, m, 5)
        res2 = f(ddf.a, m, 5, split_every=2)
        sol = f(df.a, m, 5)
        assert_eq(res, sol)
        assert_eq(res2, sol)
        assert res._name != res2._name