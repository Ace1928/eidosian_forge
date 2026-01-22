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
def test_dataframe_picklable():
    from pickle import dumps, loads
    from cloudpickle import dumps as cp_dumps
    from cloudpickle import loads as cp_loads
    d = _compat.makeTimeDataFrame()
    df = dd.from_pandas(d, npartitions=3)
    df = df + 2
    df2 = loads(dumps(df))
    assert_eq(df, df2)
    df2 = cp_loads(cp_dumps(df))
    assert_eq(df, df2)
    a2 = loads(dumps(df.A))
    assert_eq(df.A, a2)
    a2 = cp_loads(cp_dumps(df.A))
    assert_eq(df.A, a2)
    i2 = loads(dumps(df.index))
    assert_eq(df.index, i2)
    i2 = cp_loads(cp_dumps(df.index))
    assert_eq(df.index, i2)
    s = df.A.sum()
    s2 = cp_loads(cp_dumps(s))
    assert_eq(s, s2)