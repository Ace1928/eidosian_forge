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
def test_map_partition_sparse():
    sparse = pytest.importorskip('sparse')
    pytest.importorskip('numba', minversion='0.40.0')
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [6.0, 7.0, 8.0, 9.0, 10.0]}, index=['a', 'b', 'c', 'd', 'e'])
    ddf = dd.from_pandas(df, npartitions=2)

    def f(d):
        return sparse.COO(np.array(d))
    for pre in [lambda a: a, lambda a: a.x]:
        expected = f(pre(df))
        result = pre(ddf).map_partitions(f)
        assert isinstance(result, da.Array)
        computed = result.compute()
        assert (computed.data == expected.data).all()
        assert (computed.coords == expected.coords).all()