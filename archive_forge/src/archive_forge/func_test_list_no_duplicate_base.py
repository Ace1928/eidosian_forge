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
def test_list_no_duplicate_base(self):
    arr = pa.array([[1, 2], [3, 4, 5], None, [6, None], [7, 8]])
    chunked_arr = pa.chunked_array([arr.slice(0, 3), arr.slice(3, 1)])
    np_arr = chunked_arr.to_numpy()
    expected = np.array([[1.0, 2.0], [3.0, 4.0, 5.0], None, [6.0, np.nan]], dtype='object')
    for left, right in zip(np_arr, expected):
        if right is None:
            assert left == right
        else:
            npt.assert_array_equal(left, right)
    expected_base = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan]])
    npt.assert_array_equal(np_arr[0].base, expected_base)
    np_arr_sliced = chunked_arr.slice(1, 3).to_numpy()
    expected = np.array([[3, 4, 5], None, [6, np.nan]], dtype='object')
    for left, right in zip(np_arr_sliced, expected):
        if right is None:
            assert left == right
        else:
            npt.assert_array_equal(left, right)
    expected_base = np.array([[3.0, 4.0, 5.0, 6.0, np.nan]])
    npt.assert_array_equal(np_arr_sliced[0].base, expected_base)