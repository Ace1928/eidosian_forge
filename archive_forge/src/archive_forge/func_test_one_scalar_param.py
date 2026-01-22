import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_one_scalar_param(self):
    solver = self._get_solver(f1, jac1)
    omega = 1.0
    solver.set_f_params(omega)
    if self.solver_uses_jac:
        solver.set_jac_params(omega)
    self._check_solver(solver)