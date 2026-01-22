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
@PYARROW_MARK
def test_append_with_partition(tmpdir):
    tmp = str(tmpdir)
    df0 = pd.DataFrame({'lat': np.arange(0, 10, dtype='int64'), 'lon': np.arange(10, 20, dtype='int64'), 'value': np.arange(100, 110, dtype='int64')})
    df0.index.name = 'index'
    df1 = pd.DataFrame({'lat': np.arange(10, 20, dtype='int64'), 'lon': np.arange(10, 20, dtype='int64'), 'value': np.arange(120, 130, dtype='int64')})
    df1.index.name = 'index'
    df0['lat'] = df0['lat'].astype('Int64')
    df1.loc[df1.index[0], 'lat'] = np.nan
    df1['lat'] = df1['lat'].astype('Int64')
    dd_df0 = dd.from_pandas(df0, npartitions=1)
    dd_df1 = dd.from_pandas(df1, npartitions=1)
    dd.to_parquet(dd_df0, tmp, partition_on=['lon'], engine='pyarrow')
    dd.to_parquet(dd_df1, tmp, partition_on=['lon'], append=True, ignore_divisions=True, engine='pyarrow')
    out = dd.read_parquet(tmp, engine='pyarrow', index='index', calculate_divisions=True).compute()
    out['lon'] = out.lon.astype('int64')
    assert_eq(out.sort_values('value'), pd.concat([df0, df1])[out.columns], check_index=False)