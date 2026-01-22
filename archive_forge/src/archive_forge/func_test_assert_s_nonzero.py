import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_assert_s_nonzero():
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    obj = perspective(x + 1, s)
    prob = cp.Problem(cp.Minimize(obj), [x >= 3.14])
    with pytest.raises(AssertionError, match='pass in a recession function'):
        prob.solve()