import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_scalar_x():
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    obj = perspective(x - 1, s)
    prob = cp.Problem(cp.Minimize(obj), [x >= 3.14, s <= 1])
    prob.solve()
    assert np.isclose(prob.value, 3.14 - 1)