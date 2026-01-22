import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
@pytest.mark.skipif(np.intp(0).itemsize < 8, reason='Unhandled 32-bit GCC FP bug')
@pytest.mark.slow
@pytest.mark.parametrize(['options', 'msg'], [({'time_limit': 0.1}, _msg_time), ({'node_limit': 1}, _msg_iter)])
def test_milp_timeout_16545(options, msg):
    rng = np.random.default_rng(5123833489170494244)
    A = rng.integers(0, 5, size=(100, 100))
    b_lb = np.full(100, fill_value=-np.inf)
    b_ub = np.full(100, fill_value=25)
    constraints = LinearConstraint(A, b_lb, b_ub)
    variable_lb = np.zeros(100)
    variable_ub = np.ones(100)
    variable_bounds = Bounds(variable_lb, variable_ub)
    integrality = np.ones(100)
    c_vector = -np.ones(100)
    res = milp(c_vector, integrality=integrality, bounds=variable_bounds, constraints=constraints, options=options)
    assert res.message.startswith(msg)
    assert res['x'] is not None
    x = res['x']
    tol = 1e-08
    assert np.all(b_lb - tol <= A @ x) and np.all(A @ x <= b_ub + tol)
    assert np.all(variable_lb - tol <= x) and np.all(x <= variable_ub + tol)
    assert np.allclose(x, np.round(x))