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
def test_map_partitions_partition_info():

    def f(df, partition_info=None):
        assert partition_info is not None
        assert 'number' in partition_info
        assert 'division' in partition_info
        assert dsk['x', partition_info['number']].equals(df)
        assert dsk['x', d.divisions.index(partition_info['division'])].equals(df)
        return df
    df = d.map_partitions(f, meta=d)
    if not DASK_EXPR_ENABLED:
        layer = hlg_layer(df.dask, 'f-')
        assert not layer.is_materialized()
        df.dask.validate()
    result = df.compute(scheduler='single-threaded')
    assert type(result) == pd.DataFrame