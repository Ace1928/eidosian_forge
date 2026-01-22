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
def test_rename_columns():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7], 'b': [7, 6, 5, 4, 3, 2, 1]})
    ddf = dd.from_pandas(df, 2)
    ddf.columns = ['x', 'y']
    df.columns = ['x', 'y']
    tm.assert_index_equal(ddf.columns, pd.Index(['x', 'y']))
    tm.assert_index_equal(ddf._meta.columns, pd.Index(['x', 'y']))
    assert_eq(ddf, df)
    msg = 'Length mismatch: Expected axis has 2 elements, new values have 4 elements'
    with pytest.raises(ValueError) as err:
        ddf.columns = [1, 2, 3, 4]
    assert msg in str(err.value)
    df = pd.DataFrame({('A', '0'): [1, 2, 2, 3], ('B', 1): [1, 2, 3, 4]})
    ddf = dd.from_pandas(df, npartitions=2)
    df.columns = ['x', 'y']
    ddf.columns = ['x', 'y']
    tm.assert_index_equal(ddf.columns, pd.Index(['x', 'y']))
    tm.assert_index_equal(ddf._meta.columns, pd.Index(['x', 'y']))
    assert_eq(ddf, df)