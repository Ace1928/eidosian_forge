import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_large_integers(self):
    uint64_max = 2 ** 64 - 1
    arr = np.full(5, uint64_max, dtype=np.uint64)
    test = np.pad(arr, 1, mode='constant', constant_values=arr.min())
    expected = np.full(7, uint64_max, dtype=np.uint64)
    assert_array_equal(test, expected)
    int64_max = 2 ** 63 - 1
    arr = np.full(5, int64_max, dtype=np.int64)
    test = np.pad(arr, 1, mode='constant', constant_values=arr.min())
    expected = np.full(7, int64_max, dtype=np.int64)
    assert_array_equal(test, expected)