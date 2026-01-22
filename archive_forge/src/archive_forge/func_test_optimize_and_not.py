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
def test_optimize_and_not(tmpdir, engine):
    path = os.path.join(tmpdir, 'path.parquet')
    df = pd.DataFrame({'a': [3, 4, 2], 'b': [1, 2, 4], 'c': [5, 4, 2], 'd': [1, 2, 3]}, index=['a', 'b', 'c'])
    df.to_parquet(path, engine=engine)
    df2 = dd.read_parquet(path, engine=engine)
    df2a = df2['a'].groupby(df2['c']).first().to_delayed()
    df2b = df2['b'].groupby(df2['c']).first().to_delayed()
    df2c = df2[['a', 'b']].rolling(2).max().to_delayed()
    df2d = df2.rolling(2).max().to_delayed()
    result, = dask.compute(df2a + df2b + df2c + df2d)
    expected = [dask.compute(df2a)[0][0], dask.compute(df2b)[0][0], dask.compute(df2c)[0][0], dask.compute(df2d)[0][0]]
    for a, b in zip(result, expected):
        assert_eq(a, b)