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
def maximize_problem(self, solver) -> None:
    A = np.random.randn(5, 2)
    A = np.maximum(A, 0)
    b = np.random.randn(5)
    b = np.maximum(b, 0)
    p = Problem(Maximize(-sum(self.x)), [self.x >= 0, A @ self.x <= b])
    self.solve_QP(p, solver)
    for var in p.variables():
        self.assertItemsAlmostEqual([0.0, 0.0], var.value, places=3)