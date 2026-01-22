import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_odeint_banded_jacobian():

    def func(y, t, c):
        return c.dot(y)

    def jac(y, t, c):
        return c

    def jac_transpose(y, t, c):
        return c.T.copy(order='C')

    def bjac_rows(y, t, c):
        jac = np.vstack((np.r_[0, np.diag(c, 1)], np.diag(c), np.r_[np.diag(c, -1), 0], np.r_[np.diag(c, -2), 0, 0]))
        return jac

    def bjac_cols(y, t, c):
        return bjac_rows(y, t, c).T.copy(order='C')
    c = array([[-205, 0.01, 0.0, 0.0], [0.1, -2.5, 0.02, 0.0], [0.001, 0.01, -2.0, 0.01], [0.0, 0.0, 0.1, -1.0]])
    y0 = np.ones(4)
    t = np.array([0, 5, 10, 100])
    sol1, info1 = odeint(func, y0, t, args=(c,), full_output=True, atol=1e-13, rtol=1e-11, mxstep=10000, Dfun=jac)
    sol2, info2 = odeint(func, y0, t, args=(c,), full_output=True, atol=1e-13, rtol=1e-11, mxstep=10000, Dfun=jac_transpose, col_deriv=True)
    sol3, info3 = odeint(func, y0, t, args=(c,), full_output=True, atol=1e-13, rtol=1e-11, mxstep=10000, Dfun=bjac_rows, ml=2, mu=1)
    sol4, info4 = odeint(func, y0, t, args=(c,), full_output=True, atol=1e-13, rtol=1e-11, mxstep=10000, Dfun=bjac_cols, ml=2, mu=1, col_deriv=True)
    assert_allclose(sol1, sol2, err_msg='sol1 != sol2')
    assert_allclose(sol1, sol3, atol=1e-12, err_msg='sol1 != sol3')
    assert_allclose(sol3, sol4, err_msg='sol3 != sol4')
    assert_array_equal(info1['nje'], info2['nje'])
    assert_array_equal(info3['nje'], info4['nje'])
    sol1ty, info1ty = odeint(lambda t, y, c: func(y, t, c), y0, t, args=(c,), full_output=True, atol=1e-13, rtol=1e-11, mxstep=10000, Dfun=lambda t, y, c: jac(y, t, c), tfirst=True)
    assert_allclose(sol1, sol1ty, rtol=1e-12, err_msg='sol1 != sol1ty')