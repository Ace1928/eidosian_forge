import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def pcp_1() -> SolverTestHelper:
    """
    Use a 3D power cone formulation for

    min 3 * x[0] + 2 * x[1] + x[2]
    s.t. norm(x,2) <= y
         x[0] + x[1] + 3*x[2] >= 1.0
         y <= 5
    """
    x = cp.Variable(shape=(3,))
    y_square = cp.Variable()
    epis = cp.Variable(shape=(3,))
    constraints = [cp.constraints.PowCone3D(np.ones(3), epis, x, cp.Constant([0.5, 0.5, 0.5])), cp.sum(epis) <= y_square, x[0] + x[1] + 3 * x[2] >= 1.0, y_square <= 25]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
    expect_x = np.array([-3.874621860638774, -2.129788233677883, 2.33480343377204])
    expect_epis = expect_x ** 2
    expect_x = np.round(expect_x, decimals=5)
    expect_epis = np.round(expect_epis, decimals=5)
    expect_y_square = 25
    var_pairs = [(x, expect_x), (epis, expect_epis), (y_square, expect_y_square)]
    expect_ineq1 = 0.7793969212001993
    expect_ineq2 = 2.865602615049077 / 10
    expect_pc = [np.array([4.30209047, 1.29985494, 1.56211543]), np.array([0.28655796, 0.28655796, 0.28655796]), np.array([2.22062898, 1.22062899, -1.33811302])]
    con_pairs = [(constraints[0], expect_pc), (constraints[1], expect_ineq2), (constraints[2], expect_ineq1), (constraints[3], expect_ineq2)]
    obj_pair = (obj, -13.548638904065102)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth