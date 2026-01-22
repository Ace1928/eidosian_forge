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
def test_to_pandas_multiple_chunks(self):
    gc.collect()
    bytes_start = pa.total_allocated_bytes()
    ints1 = pa.array([1], type=pa.int64())
    ints2 = pa.array([2], type=pa.int64())
    arr1 = pa.StructArray.from_arrays([ints1], ['ints'])
    arr2 = pa.StructArray.from_arrays([ints2], ['ints'])
    arr = pa.chunked_array([arr1, arr2])
    expected = pd.Series([{'ints': 1}, {'ints': 2}])
    series = pd.Series(arr.to_pandas())
    tm.assert_series_equal(series, expected)
    del series
    del arr
    del arr1
    del arr2
    del ints1
    del ints2
    bytes_end = pa.total_allocated_bytes()
    assert bytes_end == bytes_start