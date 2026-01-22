import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_dlsim_trivial(self):
    a = np.array([[0.0]])
    b = np.array([[0.0]])
    c = np.array([[0.0]])
    d = np.array([[0.0]])
    n = 5
    u = np.zeros(n).reshape(-1, 1)
    tout, yout, xout = dlsim((a, b, c, d, 1), u)
    assert_array_equal(tout, np.arange(float(n)))
    assert_array_equal(yout, np.zeros((n, 1)))
    assert_array_equal(xout, np.zeros((n, 1)))