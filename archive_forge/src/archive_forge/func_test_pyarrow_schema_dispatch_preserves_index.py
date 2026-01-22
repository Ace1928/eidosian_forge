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
@pytest.mark.parametrize('preserve_index', [True, False])
def test_pyarrow_schema_dispatch_preserves_index(preserve_index):
    from dask.dataframe.dispatch import pyarrow_schema_dispatch, to_pyarrow_table_dispatch
    pytest.importorskip('pyarrow')
    df = pd.DataFrame(np.random.randn(10, 3), columns=list('abc'))
    df['d'] = pd.Series(['cat', 'dog'] * 5, dtype='string[pyarrow]')
    table = to_pyarrow_table_dispatch(df, preserve_index=preserve_index)
    schema = pyarrow_schema_dispatch(df, preserve_index=preserve_index)
    assert schema.equals(table.schema)