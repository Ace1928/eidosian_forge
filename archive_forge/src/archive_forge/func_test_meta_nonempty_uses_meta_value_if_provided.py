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
@pytest.mark.skip_with_pyarrow_strings
def test_meta_nonempty_uses_meta_value_if_provided():
    base = pd.Series([1, 2, 3], dtype='datetime64[ns]')
    offsets = pd.Series([pd.offsets.DateOffset(years=o) for o in range(3)])
    dask_base = dd.from_pandas(base, npartitions=1)
    dask_offsets = dd.from_pandas(offsets, npartitions=1)
    dask_offsets._meta = offsets.head()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PerformanceWarning)
        warnings.simplefilter('ignore', UserWarning)
        expected = base + offsets
        actual = dask_base + dask_offsets
        assert_eq(expected, actual)