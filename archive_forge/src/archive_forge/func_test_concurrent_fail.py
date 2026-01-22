import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_concurrent_fail(self):
    for sol in ('vode', 'zvode', 'lsoda'):

        def f(t, y):
            return 1.0
        r = ode(f).set_integrator(sol)
        r.set_initial_value(0, 0)
        r2 = ode(f).set_integrator(sol)
        r2.set_initial_value(0, 0)
        r.integrate(r.t + 0.1)
        r2.integrate(r2.t + 0.1)
        assert_raises(RuntimeError, r.integrate, r.t + 0.1)