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
def test_array_from_strided_bool():
    arr = np.ones((3, 2), dtype=bool)
    result = pa.array(arr[:, 0])
    expected = pa.array([True, True, True])
    assert result.equals(expected)
    result = pa.array(arr[0, :])
    expected = pa.array([True, True])
    assert result.equals(expected)