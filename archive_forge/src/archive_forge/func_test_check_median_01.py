import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_check_median_01(self):
    a = np.array([[3, 1, 4], [4, 5, 9], [9, 8, 2]])
    a = np.pad(a, 1, 'median')
    b = np.array([[4, 4, 5, 4, 4], [3, 3, 1, 4, 3], [5, 4, 5, 9, 5], [8, 9, 8, 2, 8], [4, 4, 5, 4, 4]])
    assert_array_equal(a, b)