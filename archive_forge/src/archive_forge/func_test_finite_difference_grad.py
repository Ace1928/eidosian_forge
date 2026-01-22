import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize._differentiable_functions import (ScalarFunction,
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.optimize._hessian_update_strategy import BFGS
def test_finite_difference_grad(self):
    ex = ExScalarFunction()
    nfev = 0
    ngev = 0
    x0 = [1.0, 0.0]
    analit = ScalarFunction(ex.fun, x0, (), ex.grad, ex.hess, None, (-np.inf, np.inf))
    nfev += 1
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev, nfev)
    assert_array_equal(ex.ngev, ngev)
    assert_array_equal(analit.ngev, nfev)
    approx = ScalarFunction(ex.fun, x0, (), '2-point', ex.hess, None, (-np.inf, np.inf))
    nfev += 3
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    assert_array_equal(analit.f, approx.f)
    assert_array_almost_equal(analit.g, approx.g)
    x = [10, 0.3]
    f_analit = analit.fun(x)
    g_analit = analit.grad(x)
    nfev += 1
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    f_approx = approx.fun(x)
    g_approx = approx.grad(x)
    nfev += 3
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    assert_array_almost_equal(f_analit, f_approx)
    assert_array_almost_equal(g_analit, g_approx)
    x = [2.0, 1.0]
    g_analit = analit.grad(x)
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    g_approx = approx.grad(x)
    nfev += 3
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    assert_array_almost_equal(g_analit, g_approx)
    x = [2.5, 0.3]
    f_analit = analit.fun(x)
    g_analit = analit.grad(x)
    nfev += 1
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    f_approx = approx.fun(x)
    g_approx = approx.grad(x)
    nfev += 3
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    assert_array_almost_equal(f_analit, f_approx)
    assert_array_almost_equal(g_analit, g_approx)
    x = [2, 0.3]
    f_analit = analit.fun(x)
    g_analit = analit.grad(x)
    nfev += 1
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    f_approx = approx.fun(x)
    g_approx = approx.grad(x)
    nfev += 3
    ngev += 1
    assert_array_equal(ex.nfev, nfev)
    assert_array_equal(analit.nfev + approx.nfev, nfev)
    assert_array_equal(analit.ngev + approx.ngev, ngev)
    assert_array_almost_equal(f_analit, f_approx)
    assert_array_almost_equal(g_analit, g_approx)