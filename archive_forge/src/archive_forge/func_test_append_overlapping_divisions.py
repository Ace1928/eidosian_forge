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
@pytest.mark.parametrize('metadata_file', [False, True])
@pytest.mark.parametrize(('index', 'offset'), [(pd.date_range('2022-01-01', '2022-01-31', freq='D'), pd.Timedelta(days=1)), (pd.RangeIndex(0, 500, 1), 499)])
def test_append_overlapping_divisions(tmpdir, engine, metadata_file, index, offset):
    """Test raising of error when divisions overlapping."""
    tmp = str(tmpdir)
    df = pd.DataFrame({'i32': np.arange(len(index), dtype=np.int32), 'i64': np.arange(len(index), dtype=np.int64), 'f': np.arange(len(index), dtype=np.float64), 'bhello': np.random.choice(['hello', 'yo', 'people'], size=len(index)).astype('O')}, index=index)
    ddf1 = dd.from_pandas(df, chunksize=100)
    ddf2 = dd.from_pandas(df.set_index(df.index + offset), chunksize=100)
    ddf1.to_parquet(tmp, engine=engine, write_metadata_file=metadata_file)
    with pytest.raises(ValueError, match='overlap with previously written divisions'):
        ddf2.to_parquet(tmp, engine=engine, append=True)
    ddf2.to_parquet(tmp, engine=engine, append=True, ignore_divisions=True)