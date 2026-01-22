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
@pytest.mark.parametrize('use_index', [True, False])
@pytest.mark.parametrize('n', [2, 5])
@pytest.mark.parametrize('partition_size', ['1kiB', 379])
@pytest.mark.parametrize('transform', [lambda df: df, lambda df: df.x])
def test_repartition_partition_size(use_index, n, partition_size, transform):
    df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6] * 10, 'y': list('abdabd') * 10}, index=pd.Series([10, 20, 30, 40, 50, 60] * 10))
    df = transform(df)
    a = dd.from_pandas(df, npartitions=n, sort=use_index)
    b = a.repartition(partition_size=partition_size)
    assert_eq(a, b, check_divisions=False)
    assert np.all(b.map_partitions(total_mem_usage, deep=True).compute() <= 1024)
    parts = dask.get(b.dask, b.__dask_keys__())
    assert all(map(len, parts))