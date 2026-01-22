import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def lp_4() -> SolverTestHelper:
    x = cp.Variable(5)
    objective = (cp.Minimize(cp.sum(x)), np.inf)
    var_pairs = [(x, None)]
    con_pairs = [(x <= 0, None), (x >= 1, None)]
    sth = SolverTestHelper(objective, var_pairs, con_pairs)
    return sth