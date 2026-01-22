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
@pytest.mark.parametrize('narr', [np.arange(10, dtype=np.int64), np.arange(10, dtype=np.int32), np.arange(10, dtype=np.int16), np.arange(10, dtype=np.int8), np.arange(10, dtype=np.uint64), np.arange(10, dtype=np.uint32), np.arange(10, dtype=np.uint16), np.arange(10, dtype=np.uint8), np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float16)])
def test_to_numpy_roundtrip(narr):
    arr = pa.array(narr)
    assert narr.dtype == arr.to_numpy().dtype
    np.testing.assert_array_equal(narr, arr.to_numpy())
    np.testing.assert_array_equal(narr[:6], arr[:6].to_numpy())
    np.testing.assert_array_equal(narr[2:], arr[2:].to_numpy())
    np.testing.assert_array_equal(narr[2:6], arr[2:6].to_numpy())