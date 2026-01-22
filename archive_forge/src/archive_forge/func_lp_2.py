import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def lp_2() -> SolverTestHelper:
    x = cp.Variable(shape=(2,), name='x')
    objective = cp.Minimize(x[0] + 0.5 * x[1])
    constraints = [x[0] >= -100, x[0] <= -10, x[1] == 1]
    con_pairs = [(constraints[0], 1), (constraints[1], 0), (constraints[2], -0.5)]
    var_pairs = [(x, np.array([-100, 1]))]
    obj_pair = (objective, -99.5)
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth