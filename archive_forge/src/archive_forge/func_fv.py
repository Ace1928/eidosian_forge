import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def fv(t, x, omega):
    dxdt = [omega[0] * x[1], -omega[1] * x[0]]
    return dxdt