import numpy as np
import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.tests.base_test import BaseTest
def test_expcone_1(self) -> None:
    x = cp.Variable(shape=(1,))
    tempcons = [cp.exp(x[0]) <= np.exp(1), cp.exp(-x[0]) <= np.exp(1)]
    sigma = cp.suppfunc(x, tempcons)
    y = cp.Variable(shape=(1,))
    obj_expr = y[0]
    cons = [sigma(y) <= 1]
    prob = cp.Problem(cp.Minimize(obj_expr), cons)
    prob.solve(solver='ECOS')
    viol = cons[0].violation()
    self.assertLessEqual(viol, 1e-06)
    self.assertLessEqual(abs(y.value - -1), 1e-06)