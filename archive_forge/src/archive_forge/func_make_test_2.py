import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def make_test_2(ineq_form: bool):
    x = cp.Variable()
    expect_x = np.array([-1.125])
    objective = cp.Minimize(x)
    vec = [-1.125, 1, 2]
    constr1 = x >= -1.25
    constr2 = x <= 10
    constr3 = FiniteSet(x, vec, ineq_form=ineq_form)
    obj_pairs = (objective, -1.125)
    var_pairs = [(x, expect_x)]
    con_pairs = [(constr1, None), (constr2, None), (constr3, None)]
    sth = STH.SolverTestHelper(obj_pairs, var_pairs, con_pairs)
    return sth