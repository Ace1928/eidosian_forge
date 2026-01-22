import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def test_parameter_validation():
    x = [0, 1, 0.5]
    y = np.zeros((2, 3))
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y)
    x = np.linspace(0, 1, 5)
    y = np.zeros((2, 4))
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y)

    def fun(x, y, p):
        return exp_fun(x, y)

    def bc(ya, yb, p):
        return exp_bc(ya, yb)
    y = np.zeros((2, x.shape[0]))
    assert_raises(ValueError, solve_bvp, fun, bc, x, y, p=[1])

    def wrong_shape_fun(x, y):
        return np.zeros(3)
    assert_raises(ValueError, solve_bvp, wrong_shape_fun, bc, x, y)
    S = np.array([[0, 0]])
    assert_raises(ValueError, solve_bvp, exp_fun, exp_bc, x, y, S=S)