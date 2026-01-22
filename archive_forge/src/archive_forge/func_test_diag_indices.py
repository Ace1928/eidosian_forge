import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
def test_diag_indices():
    di = diag_indices(4)
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    a[di] = 100
    assert_array_equal(a, np.array([[100, 2, 3, 4], [5, 100, 7, 8], [9, 10, 100, 12], [13, 14, 15, 100]]))
    d3 = diag_indices(2, 3)
    a = np.zeros((2, 2, 2), int)
    a[d3] = 1
    assert_array_equal(a, np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]))