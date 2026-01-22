import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose, assert_equal
from numpy.lib.arraypad import _as_pairs
def test_simple_stat_length(self):
    a = np.arange(30)
    a = np.reshape(a, (6, 5))
    a = np.pad(a, ((2, 3), (3, 2)), mode='mean', stat_length=(3,))
    b = np.array([[6, 6, 6, 5, 6, 7, 8, 9, 8, 8], [6, 6, 6, 5, 6, 7, 8, 9, 8, 8], [1, 1, 1, 0, 1, 2, 3, 4, 3, 3], [6, 6, 6, 5, 6, 7, 8, 9, 8, 8], [11, 11, 11, 10, 11, 12, 13, 14, 13, 13], [16, 16, 16, 15, 16, 17, 18, 19, 18, 18], [21, 21, 21, 20, 21, 22, 23, 24, 23, 23], [26, 26, 26, 25, 26, 27, 28, 29, 28, 28], [21, 21, 21, 20, 21, 22, 23, 24, 23, 23], [21, 21, 21, 20, 21, 22, 23, 24, 23, 23], [21, 21, 21, 20, 21, 22, 23, 24, 23, 23]])
    assert_array_equal(a, b)