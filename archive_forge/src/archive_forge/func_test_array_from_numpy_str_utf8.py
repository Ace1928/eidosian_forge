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
def test_array_from_numpy_str_utf8():
    vec = np.array(['toto', 'tata'])
    vec2 = np.array(['toto', 'tata'], dtype=object)
    arr = pa.array(vec, pa.string())
    arr2 = pa.array(vec2, pa.string())
    expected = pa.array(['toto', 'tata'])
    assert arr.equals(expected)
    assert arr2.equals(expected)
    mask = np.array([False, False], dtype=bool)
    arr = pa.array(vec, pa.string(), mask=mask)
    assert arr.equals(expected)
    vec = np.array(['mañana'.encode('utf-16-le')])
    with pytest.raises(ValueError):
        pa.array(vec, pa.string())
    with pytest.raises(ValueError):
        pa.array(vec, pa.string(), mask=np.array([False]))