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
def equivalent_forms_2(self, solver) -> None:
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
    obj2 = 0.1 * (QuadForm(self.xef, P) + q.T @ self.xef + r)
    cons = [G @ self.xef == h]
    p2 = Problem(Minimize(obj2), cons)
    self.solve_QP(p2, solver)
    self.assertAlmostEqual(p2.value, 68.1119420108, places=4)