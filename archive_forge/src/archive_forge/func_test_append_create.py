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
def test_append_create(tmpdir, engine):
    """Test that appended parquet equal to the original one."""
    tmp_path = str(tmpdir)
    df = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32), 'i64': np.arange(1000, dtype=np.int64), 'f': np.arange(1000, dtype=np.float64), 'bhello': np.random.choice(['hello', 'yo', 'people'], size=1000).astype('O')})
    df.index.name = 'index'
    half = len(df) // 2
    ddf1 = dd.from_pandas(df.iloc[:half], chunksize=100)
    ddf2 = dd.from_pandas(df.iloc[half:], chunksize=100)
    ddf1.to_parquet(tmp_path, append=True, engine=engine)
    ddf2.to_parquet(tmp_path, append=True, engine=engine)
    ddf3 = dd.read_parquet(tmp_path, engine=engine)
    assert_eq(df, ddf3)