import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
def test_unique_axis_zeros(self):
    single_zero = np.empty(shape=(2, 0), dtype=np.int8)
    uniq, idx, inv, cnt = unique(single_zero, axis=0, return_index=True, return_inverse=True, return_counts=True)
    assert_equal(uniq.dtype, single_zero.dtype)
    assert_array_equal(uniq, np.empty(shape=(1, 0)))
    assert_array_equal(idx, np.array([0]))
    assert_array_equal(inv, np.array([0, 0]))
    assert_array_equal(cnt, np.array([2]))
    uniq, idx, inv, cnt = unique(single_zero, axis=1, return_index=True, return_inverse=True, return_counts=True)
    assert_equal(uniq.dtype, single_zero.dtype)
    assert_array_equal(uniq, np.empty(shape=(2, 0)))
    assert_array_equal(idx, np.array([]))
    assert_array_equal(inv, np.array([]))
    assert_array_equal(cnt, np.array([]))
    shape = (0, 2, 0, 3, 0, 4, 0)
    multiple_zeros = np.empty(shape=shape)
    for axis in range(len(shape)):
        expected_shape = list(shape)
        if shape[axis] == 0:
            expected_shape[axis] = 0
        else:
            expected_shape[axis] = 1
        assert_array_equal(unique(multiple_zeros, axis=axis), np.empty(shape=expected_shape))