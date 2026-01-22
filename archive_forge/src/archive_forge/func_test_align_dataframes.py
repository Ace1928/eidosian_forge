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
def test_align_dataframes():
    df1 = pd.DataFrame({'A': [1, 2, 3, 3, 2, 3], 'B': [1, 2, 3, 4, 5, 6]})
    df2 = pd.DataFrame({'A': [3, 1, 2], 'C': [1, 2, 3]})
    ddf1 = dd.from_pandas(df1, npartitions=2)
    ddf2 = dd.from_pandas(df2, npartitions=1)
    actual = ddf1.map_partitions(pd.merge, df2, align_dataframes=False, left_on='A', right_on='A', how='left')
    expected = pd.merge(df1, df2, left_on='A', right_on='A', how='left')
    assert_eq(actual, expected, check_index=False, check_divisions=False)
    actual = ddf2.map_partitions(pd.merge, ddf1, align_dataframes=False, left_on='A', right_on='A', how='right')
    expected = pd.merge(df2, df1, left_on='A', right_on='A', how='right')
    assert_eq(actual, expected, check_index=False, check_divisions=False)