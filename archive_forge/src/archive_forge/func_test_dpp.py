import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_dpp():
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    a = cp.Parameter()
    obj = cp.perspective(cp.square(a + x), s)
    assert not obj.is_dpp()
    obj = cp.perspective(cp.log(a + x), s)
    assert not obj.is_dpp()