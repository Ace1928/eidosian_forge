import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_copt_params(self) -> None:
    n = 10
    m = 4
    np.random.seed(0)
    A = np.random.randn(m, n)
    x = np.random.randn(n)
    y = A.dot(x)
    z = cp.Variable(n)
    objective = cp.Minimize(cp.norm1(z))
    constraints = [A @ z == y]
    problem = cp.Problem(objective, constraints)
    with self.assertRaises(AttributeError):
        problem.solve(solver=cp.COPT, invalid_kwarg=None)
    problem.solve(solver=cp.COPT, feastol=1e-09)