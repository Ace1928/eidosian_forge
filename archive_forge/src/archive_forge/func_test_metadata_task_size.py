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
@pytest.mark.parametrize('write_metadata_file', [True, False])
@pytest.mark.parametrize('metadata_task_size', [2, 0])
def test_metadata_task_size(tmpdir, engine, write_metadata_file, metadata_task_size):
    tmpdir = str(tmpdir)
    df1 = pd.DataFrame({'a': range(100), 'b': ['dog', 'cat'] * 50})
    ddf1 = dd.from_pandas(df1, npartitions=10)
    ddf1.to_parquet(path=str(tmpdir), engine=engine, write_metadata_file=write_metadata_file)
    ddf2a = dd.read_parquet(str(tmpdir), engine=engine, calculate_divisions=True)
    ddf2b = dd.read_parquet(str(tmpdir), engine=engine, calculate_divisions=True, metadata_task_size=metadata_task_size)
    assert_eq(ddf2a, ddf2b)
    with dask.config.set({'dataframe.parquet.metadata-task-size-local': metadata_task_size}):
        ddf2c = dd.read_parquet(str(tmpdir), engine=engine, calculate_divisions=True)
    assert_eq(ddf2b, ddf2c)