import numpy as np
from numpy import (arange, zeros, array, dot, sqrt, cos, sin, eye, pi, exp,
from numpy.testing import (
from pytest import raises as assert_raises
from scipy.integrate import odeint, ode, complex_ode
def test_odeint_trivial_time():
    y0 = 1
    t = [0]
    y, info = odeint(lambda y, t: -y, y0, t, full_output=True)
    assert_array_equal(y, np.array([[y0]]))