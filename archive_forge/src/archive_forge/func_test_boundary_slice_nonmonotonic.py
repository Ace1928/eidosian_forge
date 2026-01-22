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
def test_boundary_slice_nonmonotonic():
    x = np.array([-1, -2, 2, 4, 3])
    df = pd.DataFrame({'B': range(len(x))}, index=x)
    result = methods.boundary_slice(df, 0, 4)
    expected = df.iloc[2:]
    tm.assert_frame_equal(result, expected)
    result = methods.boundary_slice(df, -1, 4)
    expected = df.drop(-2)
    tm.assert_frame_equal(result, expected)
    result = methods.boundary_slice(df, -2, 3)
    expected = df.drop(4)
    tm.assert_frame_equal(result, expected)
    result = methods.boundary_slice(df, -2, 3.5)
    expected = df.drop(4)
    tm.assert_frame_equal(result, expected)
    result = methods.boundary_slice(df, -2, 4)
    expected = df
    tm.assert_frame_equal(result, expected)