from __future__ import absolute_import, division, print_function
import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_equal
import cvxpy as cp
from cvxpy.settings import EIGVAL_TOL
from cvxpy.tests.base_test import BaseTest
def test_assume_psd(self) -> None:
    """Test assume_PSD argument.
        """
    x = cp.Variable(3)
    A = np.eye(3)
    expr = cp.quad_form(x, A, assume_PSD=True)
    assert expr.is_convex()
    A = -np.eye(3)
    expr = cp.quad_form(x, A, assume_PSD=True)
    assert expr.is_convex()
    prob = cp.Problem(cp.Minimize(expr))
    with pytest.raises(cp.SolverError, match='(Workspace allocation error!)|(Setup Error \\(Error Code 4\\))'):
        prob.solve(solver=cp.OSQP)