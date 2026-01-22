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
def test_integer_with_nulls(self):
    int_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']
    num_values = 100
    null_mask = np.random.randint(0, 10, size=num_values) < 3
    expected_cols = []
    arrays = []
    for name in int_dtypes:
        values = np.random.randint(0, 100, size=num_values)
        arr = pa.array(values, mask=null_mask)
        arrays.append(arr)
        expected = values.astype('f8')
        expected[null_mask] = np.nan
        expected_cols.append(expected)
    ex_frame = pd.DataFrame(dict(zip(int_dtypes, expected_cols)), columns=int_dtypes)
    table = pa.Table.from_arrays(arrays, int_dtypes)
    result = table.to_pandas()
    tm.assert_frame_equal(result, ex_frame)