import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_constant_odd_pad_amount(self):
    arr = np.arange(30).reshape(5, 6)
    test = np.pad(arr, ((1,), (2,)), mode='constant', constant_values=3)
    expected = np.array([[3, 3, 3, 3, 3, 3, 3, 3, 3, 3], [3, 3, 0, 1, 2, 3, 4, 5, 3, 3], [3, 3, 6, 7, 8, 9, 10, 11, 3, 3], [3, 3, 12, 13, 14, 15, 16, 17, 3, 3], [3, 3, 18, 19, 20, 21, 22, 23, 3, 3], [3, 3, 24, 25, 26, 27, 28, 29, 3, 3], [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])
    assert_allclose(test, expected)