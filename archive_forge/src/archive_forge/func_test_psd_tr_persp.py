import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_psd_tr_persp():
    ref_P = cp.Variable((2, 2), PSD=True)
    obj = cp.trace(ref_P)
    constraints = [ref_P == np.eye(2)]
    ref_prob = cp.Problem(cp.Minimize(obj), constraints)
    ref_prob.solve(solver=cp.SCS)
    P = cp.Variable((2, 2), PSD=True)
    s = cp.Variable(nonneg=True)
    f = cp.trace(P)
    obj = cp.perspective(f, s)
    constraints = [P == np.eye(2), s == 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, ref_prob.value)