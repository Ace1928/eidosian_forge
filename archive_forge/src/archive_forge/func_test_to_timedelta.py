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
def test_to_timedelta():
    s = pd.Series(range(10))
    ds = dd.from_pandas(s, npartitions=2)
    assert_eq(pd.to_timedelta(s), dd.to_timedelta(ds))
    assert_eq(pd.to_timedelta(s, unit='h'), dd.to_timedelta(ds, unit='h'))
    s = pd.Series([1, 2, 'this will error'])
    ds = dd.from_pandas(s, npartitions=2)
    assert_eq(pd.to_timedelta(s, errors='coerce'), dd.to_timedelta(ds, errors='coerce'))
    s = pd.Series(['1', 2, '1 day 2 hours'])
    ds = dd.from_pandas(s, npartitions=2)
    assert_eq(pd.to_timedelta(s), dd.to_timedelta(ds))
    with pytest.raises(ValueError, match='unit must not be specified if the input contains a str'):
        dd.to_timedelta(ds, unit='s').compute()