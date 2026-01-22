import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def lp_1() -> SolverTestHelper:
    x = cp.Variable(shape=(2,), name='x')
    objective = cp.Minimize(-4 * x[0] - 5 * x[1])
    constraints = [2 * x[0] + x[1] <= 3, x[0] + 2 * x[1] <= 3, x[0] >= 0, x[1] >= 0]
    con_pairs = [(constraints[0], 1), (constraints[1], 2), (constraints[2], 0), (constraints[3], 0)]
    var_pairs = [(x, np.array([1, 1]))]
    obj_pair = (objective, -9)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth