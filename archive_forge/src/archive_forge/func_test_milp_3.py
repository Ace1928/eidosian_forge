import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_milp_3():
    c = [0, -1]
    A = [[-1, 1], [3, 2], [2, 3]]
    b_u = [1, 12, 12]
    b_l = np.full_like(b_u, -np.inf, dtype=np.float64)
    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones_like(c)
    res = milp(c=c, constraints=constraints, integrality=integrality)
    assert_allclose(res.fun, -2)
    assert np.allclose(res.x, [1, 2]) or np.allclose(res.x, [2, 2])
    res = milp(c=c, constraints=constraints)
    assert_allclose(res.fun, -2.8)
    assert_allclose(res.x, [1.8, 2.8])