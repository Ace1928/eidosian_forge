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
def test_string_binary_from_buffers():
    array = pa.array(['a', None, 'b', 'c'])
    buffers = array.buffers()
    copied = pa.StringArray.from_buffers(len(array), buffers[1], buffers[2], buffers[0], array.null_count, array.offset)
    assert copied.to_pylist() == ['a', None, 'b', 'c']
    binary_copy = pa.Array.from_buffers(pa.binary(), len(array), array.buffers(), array.null_count, array.offset)
    assert binary_copy.to_pylist() == [b'a', None, b'b', b'c']
    copied = pa.StringArray.from_buffers(len(array), buffers[1], buffers[2], buffers[0])
    assert copied.to_pylist() == ['a', None, 'b', 'c']
    sliced = array[1:]
    buffers = sliced.buffers()
    copied = pa.StringArray.from_buffers(len(sliced), buffers[1], buffers[2], buffers[0], -1, sliced.offset)
    assert copied.to_pylist() == [None, 'b', 'c']
    assert copied.null_count == 1
    sliced = array[2:]
    buffers = sliced.buffers()
    copied = pa.StringArray.from_buffers(len(sliced), buffers[1], buffers[2], None, -1, sliced.offset)
    assert copied.to_pylist() == ['b', 'c']
    assert copied.null_count == 0