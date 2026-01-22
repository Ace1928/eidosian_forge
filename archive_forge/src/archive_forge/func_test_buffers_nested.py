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
def test_buffers_nested():
    a = pa.array([[1, 2], None, [3, None, 4, 5]], type=pa.list_(pa.int64()))
    buffers = a.buffers()
    assert len(buffers) == 4
    null_bitmap = buffers[0].to_pybytes()
    assert bytearray(null_bitmap)[0] == 5
    offsets = buffers[1].to_pybytes()
    assert struct.unpack('4i', offsets) == (0, 2, 2, 6)
    null_bitmap = buffers[2].to_pybytes()
    assert bytearray(null_bitmap)[0] == 55
    values = buffers[3].to_pybytes()
    assert struct.unpack('qqq8xqq', values) == (1, 2, 3, 4, 5)
    a = pa.array([(42, None), None, (None, 43)], type=pa.struct([pa.field('a', pa.int8()), pa.field('b', pa.int16())]))
    buffers = a.buffers()
    assert len(buffers) == 5
    null_bitmap = buffers[0].to_pybytes()
    assert bytearray(null_bitmap)[0] == 5
    null_bitmap = buffers[1].to_pybytes()
    assert bytearray(null_bitmap)[0] == 3
    values = buffers[2].to_pybytes()
    assert struct.unpack('bxx', values) == (42,)
    null_bitmap = buffers[3].to_pybytes()
    assert bytearray(null_bitmap)[0] == 6
    values = buffers[4].to_pybytes()
    assert struct.unpack('4xh', values) == (43,)