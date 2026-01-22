import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def make_test_3(ineq_form: bool):
    """Case when vec.size==1"""
    x = cp.Variable()
    objective = cp.Minimize(cp.abs(x - 3))
    vec = [1]
    cons1 = FiniteSet(x, vec, ineq_form=ineq_form)
    expected_x = np.array([1.0])
    obj_pair = (objective, 2.0)
    var_pairs = [(x, expected_x)]
    con_pairs = [(cons1, None)]
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth