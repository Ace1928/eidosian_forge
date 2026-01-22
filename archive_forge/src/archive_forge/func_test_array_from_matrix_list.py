import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_array_from_matrix_list(self):
    a = self.a
    x = np.array([a, a])
    assert_equal(x.shape, [2, 2, 2])