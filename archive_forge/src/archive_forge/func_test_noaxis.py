import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
def test_noaxis(self):
    A = matrix([[1, 0], [0, 1]])
    assert_(A.sum() == matrix(2))
    assert_(A.mean() == matrix(0.5))