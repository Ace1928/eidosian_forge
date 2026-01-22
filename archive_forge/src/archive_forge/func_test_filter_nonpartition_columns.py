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
@pytest.mark.parametrize('calculate_divisions', [None, True])
def test_filter_nonpartition_columns(tmpdir, write_engine, read_engine, calculate_divisions):
    tmpdir = str(tmpdir)
    df_write = pd.DataFrame({'id': [1, 2, 3, 4] * 4, 'time': np.arange(16), 'random': np.random.choice(['cat', 'dog'], size=16)})
    ddf_write = dd.from_pandas(df_write, npartitions=4)
    ddf_write.to_parquet(tmpdir, write_index=False, partition_on=['id'], engine=write_engine)
    ddf_read = dd.read_parquet(tmpdir, index=False, engine=read_engine, calculate_divisions=calculate_divisions, filters=[('time', '<', 5)])
    df_read = ddf_read.compute()
    assert len(df_read) == len(df_read[df_read['time'] < 5])
    assert df_read['time'].max() < 5