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
def test_remove_redundancy_infeasibility(self):
    m, n = (10, 10)
    c = np.random.rand(n)
    A_eq = np.random.rand(m, n)
    b_eq = np.random.rand(m)
    A_eq[-1, :] = 2 * A_eq[-2, :]
    b_eq[-1] *= -1
    with suppress_warnings() as sup:
        sup.filter(OptimizeWarning, 'A_eq does not appear...')
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_infeasible(res)