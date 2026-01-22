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
def test_cast_date64_to_int():
    arr = pa.array(np.array([0, 1, 2], dtype='int64'), type=pa.date64())
    expected = pa.array([0, 1, 2], type='i8')
    result = arr.cast('i8')
    assert result.equals(expected)