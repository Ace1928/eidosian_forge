import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_setdiff1d_unique(self):
    a = np.array([3, 2, 1])
    b = np.array([7, 5, 2])
    expected = np.array([3, 1])
    actual = setdiff1d(a, b, assume_unique=True)
    assert_equal(actual, expected)