import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_psd_dualcone(self) -> None:
    np.random.seed(5)
    n = 3
    X = cp.Variable(shape=(n, n))
    sigma = cp.suppfunc(X, [X >> 0])
    A = np.random.randn(n, n)
    Y = cp.Variable(shape=(n, n))
    objective = cp.Minimize(cp.norm(A.ravel(order='F') + Y.flatten()))
    cons = [sigma(Y) <= 0]
    prob = cp.Problem(objective, cons)
    prob.solve(solver='SCS', eps=1e-08)
    viol = cons[0].violation()
    self.assertLessEqual(viol, 1e-06)
    eigs = np.linalg.eigh(Y.value)[0]
    self.assertLessEqual(np.max(eigs), 1e-06)