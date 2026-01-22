import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def test_qp(self) -> None:
    m = 15
    n = 10
    p = 5
    P = np.random.randn(n, n)
    P = np.matmul(P.T, P)
    q = np.random.randn(n)
    G = np.random.randn(m, n)
    h = np.matmul(G, np.random.randn(n))
    A = np.random.randn(p, n)
    b = np.random.randn(p)

    def qp():
        x = cp.Variable(n)
        cp.Problem(cp.Minimize(1 / 2 * cp.quad_form(x, P) + cp.matmul(q.T, x)), [cp.matmul(G, x) <= h, cp.matmul(A, x) == b]).get_problem_data(cp.OSQP)
    benchmark(qp, iters=1)