import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_complex_qp(self) -> None:
    """Test a QP with a complex variable.
        """
    A0 = np.array([0 + 1j, 2 - 1j])
    A1 = np.array([[2, -1 + 1j], [4 - 3j, -3 + 2j]])
    Z = cp.Variable(complex=True)
    X = cp.Variable(2)
    B = np.array([2 + 1j, -2j])
    objective = cp.Minimize(cp.sum_squares(A0 * Z + A1 @ X - B))
    prob = cp.Problem(objective)
    prob.solve(solver='SCS')
    self.assertEqual(prob.status, cp.OPTIMAL)
    constraints = [X >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver='SCS')
    self.assertEqual(prob.status, cp.OPTIMAL)
    assert constraints[0].dual_value is not None