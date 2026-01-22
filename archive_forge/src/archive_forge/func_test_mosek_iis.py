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
def test_mosek_iis(self) -> None:
    """Test IIS feature in Mosek."""
    n = 2
    x = cp.Variable(n)
    objective = cp.Minimize(cp.sum(x))
    constraints = [x[0] >= 1, x[0] <= -1, x[1] >= 3]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    iis = problem.solver_stats.extra_stats['IIS']
    assert iis[constraints[0].id] > 0
    assert iis[constraints[1].id] > 0
    assert iis[constraints[2].id] == 0
    n = 3
    m = 2
    X = cp.Variable((m, n))
    y = cp.Variable()
    objective = cp.Minimize(cp.sum(X))
    constraints = [y == 2, X >= 3, X[0, 0] + y <= -5]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)
    iis = problem.solver_stats.extra_stats['IIS']
    assert abs(iis[constraints[0].id]) > 0
    dual1 = np.reshape(iis[constraints[1].id], X.shape, order='C')
    assert dual1[0, 0] > 0
    assert dual1[0, 1] == 0
    assert np.all(dual1[1, :] == 0)
    assert iis[constraints[2].id] > 0