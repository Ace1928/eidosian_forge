import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
def test_maxiter(self):
    c = [4, 8, 3, 0, 0, 0]
    A = [[2, 5, 3, -1, 0, 0], [3, 2.5, 8, 0, -1, 0], [8, 10, 4, 0, 0, -1]]
    b = [185, 155, 600]
    np.random.seed(0)
    maxiter = 3
    res = linprog(c, A_eq=A, b_eq=b, method=self.method, options={'maxiter': maxiter})
    _assert_iteration_limit_reached(res, maxiter)
    assert_equal(res.nit, maxiter)