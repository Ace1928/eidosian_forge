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
def regression_1(self, solver) -> None:
    np.random.seed(1)
    n = 100
    true_coeffs = np.array([[2, -2, 0.5]]).T
    x_data = np.random.rand(n) * 5
    x_data = np.atleast_2d(x_data)
    x_data_expanded = np.vstack([np.power(x_data, i) for i in range(1, 4)])
    x_data_expanded = np.atleast_2d(x_data_expanded)
    y_data = x_data_expanded.T.dot(true_coeffs) + 0.5 * np.random.rand(n, 1)
    y_data = np.atleast_2d(y_data)
    line = self.offset + x_data * self.slope
    residuals = line.T - y_data
    fit_error = sum_squares(residuals)
    p = Problem(Minimize(fit_error), [])
    self.solve_QP(p, solver)
    self.assertAlmostEqual(1171.60037715, p.value, places=4)