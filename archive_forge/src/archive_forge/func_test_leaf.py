import copy
import numpy as np
import cvxpy as cp
from cvxpy.constraints import Equality
def test_leaf():
    a = cp.Variable()
    b = copy.copy(a)
    c = copy.deepcopy(a)
    assert a.id == b.id
    assert a.id != c.id
    assert id(a) == id(b)
    assert id(a) != id(c)