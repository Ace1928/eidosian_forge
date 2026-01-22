import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_socp_2() -> SolverTestHelper:
    """
    An (unnecessarily) SOCP-based reformulation of MI_LP_1.
    Doesn't use SOC objects.
    """
    x = cp.Variable(shape=(2,))
    bool_var = cp.Variable(boolean=True)
    int_var = cp.Variable(integer=True)
    objective = cp.Minimize(-4 * x[0] - 5 * x[1])
    constraints = [2 * x[0] + x[1] <= int_var, (x[0] + 2 * x[1]) ** 2 <= 9 * bool_var, x >= 0, int_var == 3 * bool_var, int_var == 3]
    obj_pair = (objective, -9)
    var_pairs = [(x, np.array([1, 1])), (bool_var, 1), (int_var, 3)]
    con_pairs = [(con, None) for con in constraints]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth