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
def test_bounds_mixed(self):
    c = np.array([-1, 4]) * -1
    A_ub = np.array([[-3, 1], [1, 2]], dtype=np.float64)
    b_ub = [6, 4]
    x0_bounds = (-np.inf, np.inf)
    x1_bounds = (-3, np.inf)
    bounds = (x0_bounds, x1_bounds)
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_fun=-80 / 7, desired_x=[-8 / 7, 18 / 7])