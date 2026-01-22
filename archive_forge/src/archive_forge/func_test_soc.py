import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_soc(self) -> None:
    """Test with SOC.
        """
    x = Variable(2, complex=True)
    t = Variable()
    prob = Problem(cp.Minimize(t), [cp.SOC(t, x), x == 2j])
    result = prob.solve(solver='SCS', eps=1e-06)
    self.assertAlmostEqual(result, 2 * np.sqrt(2))
    self.assertItemsAlmostEqual(x.value, [2j, 2j])