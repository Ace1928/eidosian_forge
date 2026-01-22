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
def smooth_ridge(self, solver) -> None:
    np.random.seed(1)
    n = 50
    k = 20
    eta = 1
    A = np.ones((k, n))
    b = np.ones(k)
    obj = sum_squares(A @ self.xsr - b) + eta * sum_squares(self.xsr[:-1] - self.xsr[1:])
    p = Problem(Minimize(obj), [])
    self.solve_QP(p, solver)
    self.assertAlmostEqual(0, p.value, places=4)