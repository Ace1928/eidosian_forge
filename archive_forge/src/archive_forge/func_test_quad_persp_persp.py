import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_quad_persp_persp(quad_example):
    ref_val, ref_s, ref_x, r = quad_example
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    t = cp.Variable(nonneg=True)
    f_exp = cp.square(x) + r * x - 4
    obj_inner = cp.perspective(f_exp, s)
    obj = cp.perspective(obj_inner, t)
    constraints = [0.1 <= s, s <= 0.5, x >= 2, 0.1 <= t, t <= 0.5]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose=True)
    assert np.isclose(prob.value, ref_val)
    assert np.isclose(x.value, ref_x)
    assert np.isclose(s.value, ref_s)