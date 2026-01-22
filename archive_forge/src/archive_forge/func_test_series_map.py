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
@pytest.mark.parametrize('base_npart', [1, 4])
@pytest.mark.parametrize('map_npart', [1, 3])
@pytest.mark.parametrize('sorted_index', [False, True])
@pytest.mark.parametrize('sorted_map_index', [False, True])
def test_series_map(base_npart, map_npart, sorted_index, sorted_map_index):
    if DASK_EXPR_ENABLED and map_npart != base_npart:
        pytest.xfail(reason='not yet implemented')
    base = pd.Series([''.join(np.random.choice(['a', 'b', 'c'], size=3)) for x in range(100)])
    if not sorted_index:
        index = np.arange(100)
        np.random.shuffle(index)
        base.index = index
    map_index = [''.join(x) for x in product('abc', repeat=3)]
    mapper = pd.Series(np.random.randint(50, size=len(map_index)), index=map_index)
    if not sorted_map_index:
        map_index = np.array(map_index)
        np.random.shuffle(map_index)
        mapper.index = map_index
    expected = base.map(mapper)
    dask_base = dd.from_pandas(base, npartitions=base_npart, sort=False)
    dask_map = dd.from_pandas(mapper, npartitions=map_npart, sort=False)
    result = dask_base.map(dask_map)
    assert_eq(expected, result)