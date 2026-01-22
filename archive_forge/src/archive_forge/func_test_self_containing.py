import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_self_containing(self):
    arr0d = np.array(None)
    arr0d[()] = arr0d
    assert_equal(repr(arr0d), 'array(array(..., dtype=object), dtype=object)')
    arr0d[()] = 0
    arr1d = np.array([None, None])
    arr1d[1] = arr1d
    assert_equal(repr(arr1d), 'array([None, array(..., dtype=object)], dtype=object)')
    arr1d[1] = 0
    first = np.array(None)
    second = np.array(None)
    first[()] = second
    second[()] = first
    assert_equal(repr(first), 'array(array(array(..., dtype=object), dtype=object), dtype=object)')
    first[()] = 0