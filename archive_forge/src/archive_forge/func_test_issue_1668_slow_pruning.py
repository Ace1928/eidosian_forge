import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def test_issue_1668_slow_pruning(self) -> None:
    """Regression test for https://github.com/cvxpy/cvxpy/issues/1668

        Pruning matrices caused order-of-magnitude slow downs in compilation times.
        """
    s = 2000
    t = 10
    x = np.linspace(-100.0, 100.0, s)
    rows = 50
    var = cp.Variable(shape=(rows, t))
    cost = cp.sum_squares(var @ np.tile(np.array([x]), t).reshape((t, x.shape[0])) - np.tile(x, rows).reshape((rows, s)))
    objective = cp.Minimize(cost)
    problem = cp.Problem(objective)
    start = time.time()
    problem.get_problem_data(cp.ECOS, verbose=True)
    end = time.time()
    print('Issue #1668 regression test')
    print('Compilation time: ', end - start)