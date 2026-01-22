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
def test_select_filtered_column_no_stats(tmp_path, engine):
    df = pd.DataFrame({'a': range(10), 'b': ['cat'] * 10})
    path = tmp_path / 'test_select_filtered_column_no_stats.parquet'
    if engine == 'fastparquet':
        df.to_parquet(path, stats=False, engine='fastparquet')
    else:
        df.to_parquet(path, write_statistics=False)
    ddf = dd.read_parquet(path, engine=engine, filters=[('b', '==', 'cat')])
    assert_eq(df, ddf)
    ddf = dd.read_parquet(path, engine=engine, filters=[('b', 'is not', None)])
    assert_eq(df, ddf)