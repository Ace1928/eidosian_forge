import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_setdiff1d(self):
    a = np.array([6, 5, 4, 7, 1, 2, 7, 4])
    b = np.array([2, 4, 3, 3, 2, 1, 5])
    ec = np.array([6, 7])
    c = setdiff1d(a, b)
    assert_array_equal(c, ec)
    a = np.arange(21)
    b = np.arange(19)
    ec = np.array([19, 20])
    c = setdiff1d(a, b)
    assert_array_equal(c, ec)
    assert_array_equal([], setdiff1d([], []))
    a = np.array((), np.uint32)
    assert_equal(setdiff1d(a, []).dtype, np.uint32)