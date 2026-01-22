import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_quad_quad():
    ref_x = cp.Variable()
    ref_s = cp.Variable(nonneg=True)
    obj = cp.quad_over_lin(cp.quad_over_lin(ref_x, ref_s), ref_s)
    constraints = [ref_x >= 5, ref_s <= 3]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)
    ref_prob.solve(solver=cp.ECOS)
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f = cp.power(x, 4)
    obj = cp.perspective(f, s)
    constraints = [x >= 5, s <= 3]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)
    assert np.isclose(prob.value, ref_prob.value)
    assert np.isclose(x.value, ref_x.value)
    assert np.isclose(s.value, ref_s.value)