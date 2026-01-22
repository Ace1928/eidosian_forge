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
def quad_form_bound(self, solver) -> None:
    P = np.array([[13, 12, -2], [12, 17, 6], [-2, 6, 12]])
    q = np.array([[-22], [-14.5], [13]])
    r = 1
    y_star = np.array([[1], [0.5], [-1]])
    p = Problem(Minimize(0.5 * QuadForm(self.y, P) + q.T @ self.y + r), [self.y >= -1, self.y <= 1])
    self.solve_QP(p, solver)
    for var in p.variables():
        self.assertItemsAlmostEqual(y_star, var.value, places=4)