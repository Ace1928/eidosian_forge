from __future__ import annotations
import contextlib
import glob
import math
import os
import sys
import warnings
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask
import dask.dataframe as dd
import dask.multiprocessing
from dask.array.numpy_compat import NUMPY_GE_124
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import (
from dask.dataframe.io.parquet.core import get_engine
from dask.dataframe.io.parquet.utils import _parse_pandas_metadata
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, pyarrow_strings_enabled
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key
from dask.utils_test import hlg_layer
def test_partition_on_cats_2(tmpdir, engine):
    tmp = str(tmpdir)
    d = pd.DataFrame({'a': np.random.rand(50), 'b': np.random.choice(['x', 'y', 'z'], size=50), 'c': np.random.choice(['x', 'y', 'z'], size=50)})
    d = dd.from_pandas(d, 2)
    d.to_parquet(tmp, partition_on=['b', 'c'], engine=engine)
    df = dd.read_parquet(tmp, engine=engine)
    assert set(df.b.cat.categories) == {'x', 'y', 'z'}
    assert set(df.c.cat.categories) == {'x', 'y', 'z'}
    df = dd.read_parquet(tmp, columns=['a', 'c'], engine=engine)
    assert set(df.c.cat.categories) == {'x', 'y', 'z'}
    assert 'b' not in df.columns
    assert_eq(df, df.compute())
    df = dd.read_parquet(tmp, index='c', engine=engine)
    assert set(df.index.categories) == {'x', 'y', 'z'}
    assert 'c' not in df.columns
    df = dd.read_parquet(tmp, columns='b', engine=engine)
    assert set(df.cat.categories) == {'x', 'y', 'z'}