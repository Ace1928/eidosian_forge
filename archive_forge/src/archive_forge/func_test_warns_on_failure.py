import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_warns_on_failure(self):
    solver = self._get_solver(f, jac)
    solver.set_integrator(self.solver_name, nsteps=1)
    ic = [1.0, 0.0]
    solver.set_initial_value(ic, 0.0)
    assert_warns(UserWarning, solver.integrate, pi)