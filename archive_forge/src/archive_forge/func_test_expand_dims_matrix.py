import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_expand_dims_matrix(self):
    a = np.arange(10).reshape((2, 5)).view(np.matrix)
    expanded = np.expand_dims(a, axis=1)
    assert_equal(expanded.ndim, 3)
    assert_(not isinstance(expanded, np.matrix))