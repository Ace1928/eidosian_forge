import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def make_test_6(ineq_form: bool):
    """vec contains only real quantities + passed expression is affine"""
    x = cp.Variable()
    expect_x = np.array([-1.0625])
    objective = cp.Minimize(x)
    vec = [-1.125, 1.5, 2.24]
    constr1 = x >= -1.25
    constr2 = x <= 10
    expr = 2 * x + 1
    constr3 = FiniteSet(expr, vec, ineq_form=ineq_form)
    obj_pairs = (objective, -1.0625)
    var_pairs = [(x, expect_x)]
    con_pairs = [(constr1, None), (constr2, None), (constr3, None)]
    sth = STH.SolverTestHelper(obj_pairs, var_pairs, con_pairs)
    return sth