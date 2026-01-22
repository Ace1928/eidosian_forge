import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def test_cone_matrix_stuffing_with_many_constraints(self) -> None:
    self.skipTest('This benchmark takes too long.')
    m = 2000
    n = 2000
    A = np.random.randn(m, n)
    C = np.random.rand(m // 2)
    b = np.random.randn(m)
    x = cp.Variable(n)
    cost = cp.sum(A @ x)
    constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
    constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])
    problem = cp.Problem(cp.Minimize(cost), constraints)

    def cone_matrix_stuffing_with_many_constraints():
        ConeMatrixStuffing().apply(problem)
    benchmark(cone_matrix_stuffing_with_many_constraints, iters=1)