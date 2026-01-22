import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_eigval_atoms(self) -> None:
    """Test eigenvalue atoms.
        """
    P = np.arange(9) - 2j * np.arange(9)
    P = np.reshape(P, (3, 3))
    P1 = np.conj(P.T).dot(P) / 10 + np.eye(3) * 0.1
    P2 = np.array([[10, 1j, 0], [-1j, 10, 0], [0, 0, 1]])
    for P in [P1, P2]:
        value = cp.lambda_max(P).value
        X = Variable(P.shape, complex=True)
        prob = Problem(cp.Minimize(cp.lambda_max(X)), [X == P])
        result = prob.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, value, places=2)
        eigs = np.linalg.eigvals(P).real
        value = cp.sum_largest(eigs, 2).value
        X = Variable(P.shape, complex=True)
        prob = Problem(cp.Minimize(cp.lambda_sum_largest(X, 2)), [X == P])
        result = prob.solve(solver=cp.SCS, eps=1e-08)
        self.assertAlmostEqual(result, value, places=3)
        self.assertItemsAlmostEqual(X.value, P, places=3)
        value = cp.sum_smallest(eigs, 2).value
        X = Variable(P.shape, complex=True)
        prob = Problem(cp.Maximize(cp.lambda_sum_smallest(X, 2)), [X == P])
        result = prob.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, value, places=3)