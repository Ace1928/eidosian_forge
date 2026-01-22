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
def test_append_wo_index(tmpdir, engine, metadata_file):
    """Test append with write_index=False."""
    if DASK_EXPR_ENABLED and metadata_file:
        pytest.xfail("doesn't work yet")
    tmp = str(tmpdir.join('tmp1.parquet'))
    df = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32), 'i64': np.arange(1000, dtype=np.int64), 'f': np.arange(1000, dtype=np.float64), 'bhello': np.random.choice(['hello', 'yo', 'people'], size=1000).astype('O')})
    half = len(df) // 2
    ddf1 = dd.from_pandas(df.iloc[:half], chunksize=100)
    ddf2 = dd.from_pandas(df.iloc[half:], chunksize=100)
    ddf1.to_parquet(tmp, engine=engine, write_metadata_file=metadata_file)
    with pytest.raises(ValueError) as excinfo:
        ddf2.to_parquet(tmp, write_index=False, append=True, engine=engine)
    assert 'Appended columns' in str(excinfo.value)
    tmp = str(tmpdir.join('tmp2.parquet'))
    ddf1.to_parquet(tmp, write_index=False, engine=engine, write_metadata_file=metadata_file)
    ddf2.to_parquet(tmp, write_index=False, append=True, engine=engine)
    ddf3 = dd.read_parquet(tmp, index='f', engine=engine)
    assert_eq(df.set_index('f'), ddf3)