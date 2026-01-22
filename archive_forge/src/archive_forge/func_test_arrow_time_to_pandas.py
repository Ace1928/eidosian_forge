import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_arrow_time_to_pandas(self):
    pytimes = [time(1, 2, 3, 1356), time(4, 5, 6, 1356), time(0, 0, 0)]
    expected = np.array(pytimes[:2] + [None])
    expected_ms = np.array([x.replace(microsecond=1000) for x in pytimes[:2]] + [None])
    expected_s = np.array([x.replace(microsecond=0) for x in pytimes[:2]] + [None])
    arr = np.array([_pytime_to_micros(v) for v in pytimes], dtype='int64')
    arr = np.array([_pytime_to_micros(v) for v in pytimes], dtype='int64')
    null_mask = np.array([False, False, True], dtype=bool)
    a1 = pa.array(arr, mask=null_mask, type=pa.time64('us'))
    a2 = pa.array(arr * 1000, mask=null_mask, type=pa.time64('ns'))
    a3 = pa.array((arr / 1000).astype('i4'), mask=null_mask, type=pa.time32('ms'))
    a4 = pa.array((arr / 1000000).astype('i4'), mask=null_mask, type=pa.time32('s'))
    names = ['time64[us]', 'time64[ns]', 'time32[ms]', 'time32[s]']
    batch = pa.RecordBatch.from_arrays([a1, a2, a3, a4], names)
    for arr, expected_values in [(a1, expected), (a2, expected), (a3, expected_ms), (a4, expected_s)]:
        result_pandas = arr.to_pandas()
        assert (result_pandas.values == expected_values).all()
    df = batch.to_pandas()
    expected_df = pd.DataFrame({'time64[us]': expected, 'time64[ns]': expected, 'time32[ms]': expected_ms, 'time32[s]': expected_s}, columns=names)
    tm.assert_frame_equal(df, expected_df)