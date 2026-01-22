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
def test_cplex_params(self) -> None:
    if cp.CPLEX in INSTALLED_SOLVERS:
        n, m = (10, 4)
        np.random.seed(0)
        A = np.random.randn(m, n)
        x = np.random.randn(n)
        y = A.dot(x)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.norm1(z))
        constraints = [A @ z == y]
        problem = cp.Problem(objective, constraints)
        invalid_cplex_params = {'bogus': 'foo'}
        with self.assertRaises(ValueError):
            problem.solve(solver=cp.CPLEX, cplex_params=invalid_cplex_params)
        with self.assertRaises(ValueError):
            problem.solve(solver=cp.CPLEX, invalid_kwarg=None)
        cplex_params = {'advance': 0, 'simplex.limits.iterations': 1000, 'timelimit': 1000.0, 'workdir': '"mydir"'}
        problem.solve(solver=cp.CPLEX, cplex_params=cplex_params)