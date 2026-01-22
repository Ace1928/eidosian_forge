import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_milp_5():
    c = [-3, -2, -1]
    integrality = [0, 0, 1]
    lb = [0, 0, 0]
    ub = [np.inf, np.inf, 1]
    bounds = Bounds(lb, ub)
    A = [[1, 1, 1], [4, 2, 1]]
    b_l = [-np.inf, 12]
    b_u = [7, 12]
    constraints = LinearConstraint(A, b_l, b_u)
    res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    assert_allclose(res.fun, -12)