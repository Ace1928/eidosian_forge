import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_repeated_t_values():
    """Regression test for gh-8217."""

    def func(x, t):
        return -0.25 * x
    t = np.zeros(10)
    sol = odeint(func, [1.0], t)
    assert_array_equal(sol, np.ones((len(t), 1)))
    tau = 4 * np.log(2)
    t = [0] * 9 + [tau, 2 * tau, 2 * tau, 3 * tau]
    sol = odeint(func, [1, 2], t, rtol=1e-12, atol=1e-12)
    expected_sol = np.array([[1.0, 2.0]] * 9 + [[0.5, 1.0], [0.25, 0.5], [0.25, 0.5], [0.125, 0.25]])
    assert_allclose(sol, expected_sol)
    sol = odeint(func, [1.0], [])
    assert_array_equal(sol, np.array([], dtype=np.float64).reshape((0, 1)))
    assert_raises(ValueError, odeint, func, [1.0], [0, 1, 0.5, 0])
    assert_raises(ValueError, odeint, func, [1, 2, 3], [0, -1, -2, 3])