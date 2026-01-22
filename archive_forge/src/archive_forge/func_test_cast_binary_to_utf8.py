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
def test_cast_binary_to_utf8():
    binary_arr = pa.array([b'foo', b'bar', b'baz'], type=pa.binary())
    utf8_arr = binary_arr.cast(pa.utf8())
    expected = pa.array(['foo', 'bar', 'baz'], type=pa.utf8())
    assert utf8_arr.equals(expected)
    non_utf8_values = ['mañana'.encode('utf-16-le')]
    non_utf8_binary = pa.array(non_utf8_values)
    assert non_utf8_binary.type == pa.binary()
    with pytest.raises(ValueError):
        non_utf8_binary.cast(pa.string())
    non_utf8_all_null = pa.array(non_utf8_values, mask=np.array([True]), type=pa.binary())
    casted = non_utf8_all_null.cast(pa.string())
    assert casted.null_count == 1