import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def make_test_7(ineq_form: bool):
    """For testing vectorization of FiniteSet class"""
    x = cp.Variable(shape=(4,))
    expect_x = np.array([0.0, 7.0, 3.0, 0.0])
    vec = np.arange(10)
    objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
    constr1 = FiniteSet(x, vec, ineq_form=ineq_form)
    constr2 = x[0] + 2 * x[2] <= 700
    constr3 = 2 * x[1] - 8 * x[2] <= 0
    constr4 = x[1] - 2 * x[2] + x[3] >= 1
    constr5 = x[0] + x[1] + x[2] + x[3] == 10
    obj_pair = (objective, 13.0)
    con_pairs = [(constr1, None), (constr2, None), (constr3, None), (constr4, None), (constr5, None)]
    var_pairs = [(x, expect_x)]
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth