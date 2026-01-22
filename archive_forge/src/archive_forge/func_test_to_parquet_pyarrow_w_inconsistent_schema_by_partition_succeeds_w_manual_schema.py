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
@pytest.mark.skip_with_pyarrow_strings
def test_to_parquet_pyarrow_w_inconsistent_schema_by_partition_succeeds_w_manual_schema(tmpdir):
    in_arrays = [[0, 1, 2], [3, 4], np.nan, np.nan]
    out_arrays = [[0, 1, 2], [3, 4], None, None]
    in_strings = ['a', 'b', np.nan, np.nan]
    out_strings = ['a', 'b', None, None]
    tstamp = pd.Timestamp(1513393355, unit='s')
    in_tstamps = [tstamp, tstamp, pd.NaT, pd.NaT]
    out_tstamps = [tstamp.to_datetime64(), tstamp.to_datetime64(), np.datetime64('NaT'), np.datetime64('NaT')]
    timezone = 'US/Eastern'
    tz_tstamp = pd.Timestamp(1513393355, unit='s', tz=timezone)
    in_tz_tstamps = [tz_tstamp, tz_tstamp, pd.NaT, pd.NaT]
    out_tz_tstamps = [tz_tstamp.tz_convert(None).to_datetime64(), tz_tstamp.tz_convert(None).to_datetime64(), np.datetime64('NaT'), np.datetime64('NaT')]
    df = pd.DataFrame({'partition_column': [0, 0, 1, 1], 'arrays': in_arrays, 'strings': in_strings, 'tstamps': in_tstamps, 'tz_tstamps': in_tz_tstamps})
    ddf = dd.from_pandas(df, npartitions=2)
    schema = pa.schema([('arrays', pa.list_(pa.int64())), ('strings', pa.string()), ('tstamps', pa.timestamp('ns')), ('tz_tstamps', pa.timestamp('ns', timezone)), ('partition_column', pa.int64())])
    ddf.to_parquet(str(tmpdir), partition_on='partition_column', schema=schema)
    ddf_after_write = dd.read_parquet(str(tmpdir), calculate_divisions=False).compute().reset_index(drop=True)
    arrays_after_write = ddf_after_write.arrays.values
    for i in range(len(df)):
        assert np.array_equal(arrays_after_write[i], out_arrays[i]), type(out_arrays[i])
    tstamps_after_write = ddf_after_write.tstamps.values
    for i in range(len(df)):
        if np.isnat(tstamps_after_write[i]):
            assert np.isnat(out_tstamps[i])
        else:
            assert tstamps_after_write[i] == out_tstamps[i]
    tz_tstamps_after_write = ddf_after_write.tz_tstamps.values
    for i in range(len(df)):
        if np.isnat(tz_tstamps_after_write[i]):
            assert np.isnat(out_tz_tstamps[i])
        else:
            assert tz_tstamps_after_write[i] == out_tz_tstamps[i]
    assert np.array_equal(ddf_after_write.strings.values, out_strings)
    assert np.array_equal(ddf_after_write.partition_column, df.partition_column)