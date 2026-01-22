import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def socp_1() -> SolverTestHelper:
    """
    min 3 * x[0] + 2 * x[1] + x[2]
    s.t. norm(x,2) <= y
         x[0] + x[1] + 3*x[2] >= 1.0
         y <= 5
    """
    x = cp.Variable(shape=(3,))
    y = cp.Variable()
    soc = cp.constraints.second_order.SOC(y, x)
    constraints = [soc, x[0] + x[1] + 3 * x[2] >= 1.0, y <= 5]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
    expect_x = np.array([-3.874621860638774, -2.129788233677883, 2.33480343377204])
    expect_x = np.round(expect_x, decimals=5)
    expect_y = 5
    var_pairs = [(x, expect_x), (y, expect_y)]
    expect_soc = [np.array([2.86560262]), np.array([2.22062583, 1.22062583, -1.33812252])]
    expect_ineq1 = 0.7793969212001993
    expect_ineq2 = 2.865602615049077
    con_pairs = [(constraints[0], expect_soc), (constraints[1], expect_ineq1), (constraints[2], expect_ineq2)]
    obj_pair = (obj, -13.548638904065102)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth