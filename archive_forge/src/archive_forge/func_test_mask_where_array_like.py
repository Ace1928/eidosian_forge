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
@pytest.mark.parametrize('df,cond', [(pd.DataFrame({'x': [1, 2]}, index=[1, 2]), [[True], [False]]), (pd.DataFrame({'x': [1, 2], 'y': [3, 4]}), [[True, False], [True, False]]), (pd.DataFrame({'x': [1, 2], 'y': [3, 4]}), [[True, True], [False, False]]), (pd.DataFrame({'x': [1, 2, 3, 4], 'y': [3, 4, 5, 6]}), [[True, True], [True, True], [False, False], [False, False]]), (pd.DataFrame({'x': [1, 2, 3, 4], 'y': [3, 4, 5, 6]}), [[True, False], [True, False], [True, False], [True, False]])])
def test_mask_where_array_like(df, cond):
    """DataFrame.mask fails for single-row partitions
    https://github.com/dask/dask/issues/9848
    """
    ddf = dd.from_pandas(df, npartitions=2)
    with pytest.raises(ValueError, match='can be aligned|shape'):
        ddf.mask(cond=cond, other=5)
    with pytest.raises(ValueError, match='can be aligned|shape'):
        ddf.where(cond=cond, other=5)
    dd_cond = pd.DataFrame(cond, index=df.index, columns=df.columns)
    expected = df.mask(cond=cond, other=5)
    result = ddf.mask(cond=dd_cond, other=5)
    assert_eq(expected, result)
    expected = df.where(cond=cond, other=5)
    result = ddf.where(cond=dd_cond, other=5)
    assert_eq(expected, result)