import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_numpy_ravel_order(self):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    assert_equal(np.ravel(x), [1, 2, 3, 4, 5, 6])
    assert_equal(np.ravel(x, order='F'), [1, 4, 2, 5, 3, 6])
    assert_equal(np.ravel(x.T), [1, 4, 2, 5, 3, 6])
    assert_equal(np.ravel(x.T, order='A'), [1, 2, 3, 4, 5, 6])
    x = matrix([[1, 2, 3], [4, 5, 6]])
    assert_equal(np.ravel(x), [1, 2, 3, 4, 5, 6])
    assert_equal(np.ravel(x, order='F'), [1, 4, 2, 5, 3, 6])
    assert_equal(np.ravel(x.T), [1, 4, 2, 5, 3, 6])
    assert_equal(np.ravel(x.T, order='A'), [1, 2, 3, 4, 5, 6])