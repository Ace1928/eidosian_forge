import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises as assert_raises
from scipy.signal._arraytools import (axis_slice, axis_reverse,
def test_zero_ext(self):
    a = np.array([[1, 2, 3, 4, 5], [9, 8, 7, 6, 5]])
    zero = zero_ext(a, 2, axis=1)
    expected = np.array([[0, 0, 1, 2, 3, 4, 5, 0, 0], [0, 0, 9, 8, 7, 6, 5, 0, 0]])
    assert_array_equal(zero, expected)
    zero = zero_ext(a, 1, axis=0)
    expected = np.array([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [9, 8, 7, 6, 5], [0, 0, 0, 0, 0]])
    assert_array_equal(zero, expected)