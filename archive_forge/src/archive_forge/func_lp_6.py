import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def lp_6() -> SolverTestHelper:
    """Test LP with no constraints"""
    x = cp.Variable()
    from cvxpy.expressions.constants import Constant
    objective = cp.Maximize(Constant(0.23) * x)
    obj_pair = (objective, np.inf)
    var_pairs = [(x, None)]
    sth = SolverTestHelper(obj_pair, var_pairs, [])
    return sth