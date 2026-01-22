import numpy as np
import scipy.sparse as sp
from scipy.linalg import lstsq
import cvxpy as cp
from cvxpy import Maximize, Minimize, Parameter, Problem
from cvxpy.atoms import (
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, QP_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import StandardTestLPs
def equivalent_forms_3(self, solver) -> None:
    m = 100
    n = 80
    r = 70
    np.random.seed(1)
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    G = np.random.randn(r, n)
    h = np.random.randn(r)
    P = np.dot(A.T, A)
    q = -2 * np.dot(A.T, b)
    r = np.dot(b.T, b)
    Pinv = np.linalg.inv(P)
    obj3 = 0.1 * (matrix_frac(self.xef, Pinv) + q.T @ self.xef + r)
    cons = [G @ self.xef == h]
    p3 = Problem(Minimize(obj3), cons)
    self.solve_QP(p3, solver)
    self.assertAlmostEqual(p3.value, 68.1119420108, places=4)