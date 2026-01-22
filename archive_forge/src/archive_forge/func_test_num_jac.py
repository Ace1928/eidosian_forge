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
def test_num_jac():

    def fun(t, y):
        return np.vstack([-0.04 * y[0] + 10000.0 * y[1] * y[2], 0.04 * y[0] - 10000.0 * y[1] * y[2] - 30000000.0 * y[1] ** 2, 30000000.0 * y[1] ** 2])

    def jac(t, y):
        return np.array([[-0.04, 10000.0 * y[2], 10000.0 * y[1]], [0.04, -10000.0 * y[2] - 60000000.0 * y[1], -10000.0 * y[1]], [0, 60000000.0 * y[1], 0]])
    t = 1
    y = np.array([1, 0, 0])
    J_true = jac(t, y)
    threshold = 1e-05
    f = fun(t, y).ravel()
    J_num, factor = num_jac(fun, t, y, f, threshold, None)
    assert_allclose(J_num, J_true, rtol=1e-05, atol=1e-05)
    J_num, factor = num_jac(fun, t, y, f, threshold, factor)
    assert_allclose(J_num, J_true, rtol=1e-05, atol=1e-05)