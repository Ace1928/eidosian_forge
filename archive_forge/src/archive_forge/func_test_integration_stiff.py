from itertools import product
from numpy.testing import (assert_, assert_allclose, assert_array_less,
import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.optimize._numdiff import group_columns
from scipy.integrate import solve_ivp, RK23, RK45, DOP853, Radau, BDF, LSODA
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.common import num_jac
from scipy.integrate._ivp.base import ConstantDenseOutput
from scipy.sparse import coo_matrix, csc_matrix
@pytest.mark.slow
@pytest.mark.parametrize('method', ['Radau', 'BDF', 'LSODA'])
def test_integration_stiff(method):
    rtol = 1e-06
    atol = 1e-06
    y0 = [10000.0, 0, 0]
    tspan = [0, 100000000.0]

    def fun_robertson(t, state):
        x, y, z = state
        return [-0.04 * x + 10000.0 * y * z, 0.04 * x - 10000.0 * y * z - 30000000.0 * y * y, 30000000.0 * y * y]
    res = solve_ivp(fun_robertson, tspan, y0, rtol=rtol, atol=atol, method=method)
    assert res.nfev < 5000
    assert res.njev < 200