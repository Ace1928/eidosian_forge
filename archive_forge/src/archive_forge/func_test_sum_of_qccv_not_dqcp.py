import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_sum_of_qccv_not_dqcp(self) -> None:
    t = cp.Variable(5, pos=True)
    expr = cp.sum(cp.square(t) / t)
    self.assertFalse(expr.is_dqcp())