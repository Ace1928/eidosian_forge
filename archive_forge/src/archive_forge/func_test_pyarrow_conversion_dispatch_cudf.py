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
@pytest.mark.gpu
def test_pyarrow_conversion_dispatch_cudf():
    from dask.dataframe.dispatch import from_pyarrow_table_dispatch, to_pyarrow_table_dispatch
    cudf = pytest.importorskip('cudf')

    @to_pyarrow_table_dispatch.register(cudf.DataFrame)
    def _cudf_to_table(obj, preserve_index=True):
        return obj.to_arrow(preserve_index=preserve_index)

    @from_pyarrow_table_dispatch.register(cudf.DataFrame)
    def _table_to_cudf(obj, table, self_destruct=False):
        return obj.from_arrow(table)
    df1 = cudf.DataFrame(np.random.randn(10, 3), columns=list('abc'))
    df2 = from_pyarrow_table_dispatch(df1, to_pyarrow_table_dispatch(df1))
    assert type(df1) == type(df2)
    assert_eq(df1, df2)