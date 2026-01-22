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
def test_bug_8174(self):
    A_ub = np.array([[22714, 1008, 13380, -2713.5, -1116], [-4986, -1092, -31220, 17386.5, 684], [-4986, 0, 0, -2713.5, 0], [22714, 0, 0, 17386.5, 0]])
    b_ub = np.zeros(A_ub.shape[0])
    c = -np.ones(A_ub.shape[1])
    bounds = [(0, 1)] * A_ub.shape[1]
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'invalid value encountered')
        sup.filter(LinAlgWarning)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    if self.options.get('tol', 1e-09) < 1e-10 and self.method == 'simplex':
        _assert_unable_to_find_basic_feasible_sol(res)
    else:
        _assert_success(res, desired_fun=-2.0080717488789235, atol=1e-06)