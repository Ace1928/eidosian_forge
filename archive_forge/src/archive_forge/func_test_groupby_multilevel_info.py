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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='not compatible')
def test_groupby_multilevel_info():
    from io import StringIO
    pandas_format._put_lines = put_lines
    df = pd.DataFrame({'A': [1, 1, 2, 2], 'B': [1, 2, 3, 4], 'C': [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)
    g = ddf.groupby(['A', 'B']).sum()
    _assert_info(g.compute(), g, memory_usage=True)
    buf = StringIO()
    g.info(buf, verbose=False)
    assert buf.getvalue() == "<class 'dask.dataframe.core.DataFrame'>\nColumns: 1 entries, C to C\ndtypes: int64(1)"
    g = ddf.groupby(['A', 'B']).agg(['count', 'sum'])
    _assert_info(g.compute(), g, memory_usage=True)
    buf = StringIO()
    g.info(buf, verbose=False)
    expected = "<class 'dask.dataframe.core.DataFrame'>\nColumns: 2 entries, ('C', 'count') to ('C', 'sum')\ndtypes: int64(2)"
    assert buf.getvalue() == expected