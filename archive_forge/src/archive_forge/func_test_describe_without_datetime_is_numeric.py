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
@pytest.mark.skipif(DASK_EXPR_ENABLED, reason='no longer supported')
def test_describe_without_datetime_is_numeric():
    data = {'a': ['aaa', 'bbb', 'bbb', None, None, 'zzz'] * 2, 'c': [None, 0, 1, 2, 3, 4] * 2, 'd': [None, 0, 1] * 4, 'e': [pd.Timestamp('2017-05-09 00:00:00.006000'), pd.Timestamp('2017-05-09 00:00:00.006000'), pd.Timestamp('2017-05-09 07:56:23.858694'), pd.Timestamp('2017-05-09 05:59:58.938999'), None, None] * 2}
    df = pd.DataFrame(data)
    ddf = dd.from_pandas(df, 2)
    expected = df.describe()
    if PANDAS_GE_200:
        expected = _drop_mean(expected, 'e')
    assert_eq(ddf.describe(), expected)
    for col in ['a', 'c']:
        assert_eq(df[col].describe(), ddf[col].describe())
    if PANDAS_GE_200:
        expected = _drop_mean(df.e.describe())
        assert_eq(expected, ddf.e.describe())
        with pytest.raises(TypeError, match='datetime_is_numeric is removed in pandas>=2.0.0'):
            ddf.e.describe(datetime_is_numeric=True)
    else:
        with pytest.warns(FutureWarning, match='Treating datetime data as categorical rather than numeric in `.describe` is deprecated'):
            ddf.e.describe()