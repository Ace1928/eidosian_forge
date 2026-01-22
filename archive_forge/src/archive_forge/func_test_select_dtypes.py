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
@pytest.mark.parametrize('include, exclude', [([int], None), (None, [int]), ([np.number, get_string_dtype()], [float]), (['datetime'], None)])
def test_select_dtypes(include, exclude):
    n = 10
    df = pd.DataFrame({'cint': [1] * n, 'cstr': ['a'] * n, 'clfoat': [1.0] * n, 'cdt': pd.date_range('2016-01-01', periods=n)})
    df['cstr'] = df['cstr'].astype(get_string_dtype())
    a = dd.from_pandas(df, npartitions=2)
    result = a.select_dtypes(include=include, exclude=exclude)
    expected = df.select_dtypes(include=include, exclude=exclude)
    assert_eq(result, expected)
    assert_eq_dtypes(a, df)
    assert_eq_dtypes(result, expected)