import os
import time
import numpy as np
import pytest
import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
def test_diffcp_sdp_example(self) -> None:
    self.skipTest('This benchmark takes too long.')

    def randn_symm(n):
        A = np.random.randn(n, n)
        return (A + A.T) / 2

    def randn_psd(n):
        A = 1.0 / 10 * np.random.randn(n, n)
        return np.matmul(A, A.T)
    n = 100
    p = 100
    C = randn_psd(n)
    As = [randn_symm(n) for _ in range(p)]
    Bs = np.random.randn(p)

    def diffcp_sdp():
        X = cp.Variable((n, n), PSD=True)
        objective = cp.trace(cp.matmul(C, X))
        constraints = [cp.trace(cp.matmul(As[i], X)) == Bs[i] for i in range(p)]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.get_problem_data(cp.SCS)
    benchmark(diffcp_sdp, iters=1)