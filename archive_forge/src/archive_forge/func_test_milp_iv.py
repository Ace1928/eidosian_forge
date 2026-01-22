import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_milp_iv():
    message = '`c` must be a dense array'
    with pytest.raises(ValueError, match=message):
        milp(sparse.coo_array([0, 0]))
    message = '`c` must be a one-dimensional array of finite numbers with'
    with pytest.raises(ValueError, match=message):
        milp(np.zeros((3, 4)))
    with pytest.raises(ValueError, match=message):
        milp([])
    with pytest.raises(ValueError, match=message):
        milp(None)
    message = '`bounds` must be convertible into an instance of...'
    with pytest.raises(ValueError, match=message):
        milp(1, bounds=10)
    message = '`constraints` (or each element within `constraints`) must be'
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(1, constraints=10)
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(np.zeros(3), constraints=([[1, 2, 3]], [2, 3], [2, 3]))
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(np.zeros(2), constraints=([[1, 2]], [2], sparse.coo_array([2])))
    message = 'The shape of `A` must be (len(b_l), len(c)).'
    with pytest.raises(ValueError, match=re.escape(message)):
        milp(np.zeros(3), constraints=([[1, 2]], [2], [2]))
    message = '`integrality` must be a dense array'
    with pytest.raises(ValueError, match=message):
        milp([1, 2], integrality=sparse.coo_array([1, 2]))
    message = '`integrality` must contain integers 0-3 and be broadcastable to `c.shape`.'
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], integrality=[1, 2])
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], integrality=[1, 5, 3])
    message = 'Lower and upper bounds must be dense arrays.'
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2], sparse.coo_array([3, 4])))
    message = '`lb`, `ub`, and `keep_feasible` must be broadcastable.'
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2], [3, 4, 5]))
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2, 3], [4, 5]))
    message = '`bounds.lb` and `bounds.ub` must contain reals and...'
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2], [3, 4]))
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2, 3], ['3+4', 4, 5]))
    with pytest.raises(ValueError, match=message):
        milp([1, 2, 3], bounds=([1, 2, 3], [set(), 4, 5]))