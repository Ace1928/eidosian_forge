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
def test_fillna_dask_dataframe_input():
    df = _compat.makeMissingDataframe()
    df1 = _compat.makeMissingDataframe()
    ddf = dd.from_pandas(df, npartitions=5)
    ddf1 = dd.from_pandas(df1, npartitions=3)
    assert_eq(ddf.fillna(ddf1), df.fillna(df1))
    ddf_unknown = dd.from_pandas(df, npartitions=5, sort=False)
    if DASK_EXPR_ENABLED:
        assert_eq(ddf_unknown.fillna(ddf1), df.fillna(df1))
    else:
        with pytest.raises(ValueError, match='Not all divisions are known'):
            assert_eq(ddf_unknown.fillna(ddf1), df.fillna(df1))