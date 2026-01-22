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
def test_values_extension_dtypes():
    from dask.array.utils import assert_eq
    df = pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [2, 3, 4, 5]}, index=pd.Index([1.0, 2.0, 3.0, 4.0], dtype='Float64', name='ind'))
    df = df.astype({'x': 'string[python]', 'y': 'Int64'})
    ddf = dd.from_pandas(df, npartitions=2)
    assert_eq(df.values, ddf.values)
    with pytest.warns(UserWarning, match='object dtype'):
        result = ddf.x.values
    assert_eq(result, df.x.values.astype(object))
    with pytest.warns(UserWarning, match='object dtype'):
        result = ddf.y.values
    assert_eq(result, df.y.values.astype(object))
    ctx = contextlib.nullcontext()
    if PANDAS_GE_140:
        ctx = pytest.warns(UserWarning, match='object dtype')
    with ctx:
        result = ddf.index.values
    assert_eq(result, df.index.values.astype(object))