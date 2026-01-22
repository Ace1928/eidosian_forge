from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
@pytest.mark.parametrize(('dtype', 'type'), [('timedelta64[s]', pa.duration('s')), ('timedelta64[ms]', pa.duration('ms')), ('timedelta64[us]', pa.duration('us')), ('timedelta64[ns]', pa.duration('ns'))])
def test_array_from_numpy_timedelta(dtype, type):
    data = [None, datetime.timedelta(1), datetime.timedelta(0, 1)]
    np_arr = np.array(data, dtype=dtype)
    arr = pa.array(np_arr)
    assert isinstance(arr, pa.DurationArray)
    assert arr.type == type
    expected = pa.array(data, type=type)
    assert arr.equals(expected)
    assert arr.to_pylist() == data
    arr = pa.array(list(np.array(data, dtype=dtype)))
    assert arr.equals(expected)
    assert arr.to_pylist() == data