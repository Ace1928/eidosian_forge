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
def test_append_different_columns(tmpdir, engine, metadata_file):
    """Test raising of error when non equal columns."""
    tmp = str(tmpdir)
    df1 = pd.DataFrame({'i32': np.arange(100, dtype=np.int32)})
    df2 = pd.DataFrame({'i64': np.arange(100, dtype=np.int64)})
    df3 = pd.DataFrame({'i32': np.arange(100, dtype=np.int64)})
    ddf1 = dd.from_pandas(df1, chunksize=2)
    ddf2 = dd.from_pandas(df2, chunksize=2)
    ddf3 = dd.from_pandas(df3, chunksize=2)
    ddf1.to_parquet(tmp, engine=engine, write_metadata_file=metadata_file)
    with pytest.raises(ValueError) as excinfo:
        ddf2.to_parquet(tmp, engine=engine, append=True)
    assert 'Appended columns' in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        ddf3.to_parquet(tmp, engine=engine, append=True)
    assert 'Appended dtypes' in str(excinfo.value)