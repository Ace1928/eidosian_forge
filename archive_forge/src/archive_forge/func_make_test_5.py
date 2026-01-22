import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def make_test_5(ineq_form: bool):
    """Case when input expression to FiniteSet constraint is affine"""
    x = cp.Variable(shape=(4,))
    vec = np.arange(10)
    objective = cp.Maximize(x[0] + x[1] + 2 * x[2] - 2 * x[3])
    expr0 = 2 * x[0] + 1
    expr2 = 3 * x[2] + 5
    constr1 = FiniteSet(expr0, vec, ineq_form=ineq_form)
    constr2 = FiniteSet(x[1], vec, ineq_form=ineq_form)
    constr3 = FiniteSet(expr2, vec, ineq_form=ineq_form)
    constr4 = FiniteSet(x[3], vec, ineq_form=ineq_form)
    constr5 = x[0] + 2 * x[2] <= 700
    constr6 = 2 * x[1] - 8 * x[2] <= 0
    constr7 = x[1] - 2 * x[2] + x[3] >= 1
    constr8 = x[0] + x[1] + x[2] + x[3] == 10
    expected_x = np.array([4.0, 4.0, 1.0, 1.0])
    obj_pair = (objective, 8.0)
    con_pairs = [(constr1, None), (constr2, None), (constr3, None), (constr4, None), (constr5, None), (constr6, None), (constr7, None), (constr8, None)]
    var_pairs = [(x, expected_x)]
    sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
    return sth