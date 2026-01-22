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
def test_array_conversions_no_sentinel_values():
    arr = np.array([1, 2, 3, 4], dtype='int8')
    refcount = sys.getrefcount(arr)
    arr2 = pa.array(arr)
    assert sys.getrefcount(arr) == refcount + 1
    assert arr2.type == 'int8'
    arr3 = pa.array(np.array([1, np.nan, 2, 3, np.nan, 4], dtype='float32'), type='float32')
    assert arr3.type == 'float32'
    assert arr3.null_count == 0