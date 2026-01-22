import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import (
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.linalg import gershgorin_psd_check
def test_constant_psd_nsd(self):
    n = 5
    np.random.randn(0)
    U = np.random.randn(n, n)
    U = U @ U.T
    evals, U = np.linalg.eigh(U)
    v1 = np.array([3, 2, 1, 1e-08, -1])
    P = Constant(U @ np.diag(v1) @ U.T)
    self.assertFalse(P.is_psd())
    self.assertFalse(P.is_nsd())
    v2 = np.array([3, 2, 2, 1e-06, -1])
    P = Constant(U @ np.diag(v2) @ U.T)
    self.assertFalse(P.is_psd())
    self.assertFalse(P.is_nsd())
    v3 = np.array([3, 2, 2, 0.0001, -1e-06])
    P = Constant(U @ np.diag(v3) @ U.T)
    self.assertFalse(P.is_psd())
    self.assertFalse(P.is_nsd())
    v4 = np.array([-1, 3, 0, 0, 0])
    P = Constant(U @ np.diag(v4) @ U.T)
    self.assertFalse(P.is_psd())
    self.assertFalse(P.is_nsd())
    P = Constant(np.array([[1, 2], [2, 1]]))
    x = Variable(shape=(2,))
    expr = cp.quad_form(x, P)
    self.assertFalse(expr.is_dcp())
    self.assertFalse((-expr).is_dcp())
    self.assertFalse(gershgorin_psd_check(P.value, tol=0.99))
    P = Constant(np.array([[2, 1], [1, 2]]))
    self.assertTrue(gershgorin_psd_check(P.value, tol=0.0))
    P = Constant(np.diag(9 * [0.0001] + [-10000.0]))
    self.assertFalse(P.is_psd())
    self.assertFalse(P.is_nsd())
    P = Constant(np.ones(shape=(5, 5)))
    self.assertTrue(P.is_psd())
    self.assertFalse(P.is_nsd())
    P = Constant(sp.eye(10))
    self.assertTrue(gershgorin_psd_check(P.value, s.EIGVAL_TOL))
    self.assertTrue(P.is_psd())
    self.assertTrue((-P).is_nsd())
    Q = -s.EIGVAL_TOL / 2 * P
    self.assertTrue(gershgorin_psd_check(Q.value, s.EIGVAL_TOL))
    Q = -1.1 * s.EIGVAL_TOL * P
    self.assertFalse(gershgorin_psd_check(Q.value, s.EIGVAL_TOL))
    self.assertFalse(Q.is_psd())