import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_setxor1d(self):
    a = np.array([5, 7, 1, 2])
    b = np.array([2, 4, 3, 1, 5])
    ec = np.array([3, 4, 7])
    c = setxor1d(a, b)
    assert_array_equal(c, ec)
    a = np.array([1, 2, 3])
    b = np.array([6, 5, 4])
    ec = np.array([1, 2, 3, 4, 5, 6])
    c = setxor1d(a, b)
    assert_array_equal(c, ec)
    a = np.array([1, 8, 2, 3])
    b = np.array([6, 5, 4, 8])
    ec = np.array([1, 2, 3, 4, 5, 6])
    c = setxor1d(a, b)
    assert_array_equal(c, ec)
    assert_array_equal([], setxor1d([], []))