import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def mi_lp_3() -> SolverTestHelper:
    x = cp.Variable(4, boolean=True)
    from cvxpy.expressions.constants import Constant
    objective = cp.Maximize(Constant(1))
    constraints = [x[0] + x[1] + x[2] + x[3] <= 2, x[0] + x[1] + x[2] + x[3] >= 2, x[0] + x[1] <= 1, x[0] + x[2] <= 1, x[0] + x[3] <= 1, x[2] + x[3] <= 1, x[1] + x[3] <= 1, x[1] + x[2] <= 1]
    obj_pair = (objective, -np.inf)
    con_pairs = [(c, None) for c in constraints]
    var_pairs = [(x, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth