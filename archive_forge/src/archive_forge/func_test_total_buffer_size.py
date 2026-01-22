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
def test_total_buffer_size():
    a = pa.array(np.array([4, 5, 6], dtype='int64'))
    assert a.nbytes == 8 * 3
    assert a.get_total_buffer_size() == 8 * 3
    assert sys.getsizeof(a) >= object.__sizeof__(a) + a.nbytes
    a = pa.array([1, None, 3], type='int64')
    assert a.nbytes == 8 * 3 + 1
    assert a.get_total_buffer_size() == 8 * 3 + 1
    assert sys.getsizeof(a) >= object.__sizeof__(a) + a.nbytes
    a = pa.array([[1, 2], None, [3, None, 4, 5]], type=pa.list_(pa.int64()))
    assert a.nbytes == 62
    assert a.get_total_buffer_size() == 1 + 4 * 4 + 1 + 6 * 8
    assert sys.getsizeof(a) >= object.__sizeof__(a) + a.nbytes
    a = pa.array([[[5, 6, 7]], [[9, 10]]], type=pa.list_(pa.list_(pa.int8())))
    assert a.get_total_buffer_size() == 4 * 3 + 4 * 3 + 1 * 5
    assert a.nbytes == 21
    a = pa.array([[[1, 2], [3, 4]], [[5, 6, 7], None, [8]], [[9, 10]]], type=pa.list_(pa.list_(pa.int8())))
    a1 = a.slice(1, 2)
    assert a1.nbytes == 4 * 2 + 1 + 4 * 4 + 1 * 6
    assert a1.get_total_buffer_size() == 4 * 4 + 1 + 4 * 7 + 1 * 10