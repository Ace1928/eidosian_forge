import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_matrix_element(self):
    x = matrix([[1, 2, 3], [4, 5, 6]])
    assert_equal(x[0][0], matrix([[1, 2, 3]]))
    assert_equal(x[0][0].shape, (1, 3))
    assert_equal(x[0].shape, (1, 3))
    assert_equal(x[:, 0].shape, (2, 1))
    x = matrix(0)
    assert_equal(x[0, 0], 0)
    assert_equal(x[0], 0)
    assert_equal(x[:, 0].shape, x.shape)