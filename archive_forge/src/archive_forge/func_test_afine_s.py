import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_afine_s():
    x = cp.Variable()
    s = cp.Variable(2)
    with pytest.raises(AssertionError, match='s must be a variable'):
        perspective(cp.square(x), cp.sum(s))