import pytest
import numpy as np
from numpy.testing import TestCase, assert_array_equal
import scipy.sparse as sps
from scipy.optimize._constraints import (
def test_residual(self):
    A = np.eye(2)
    lc = LinearConstraint(A, -2, 4)
    x0 = [-1, 2]
    np.testing.assert_allclose(lc.residual(x0), ([1, 4], [5, 2]))