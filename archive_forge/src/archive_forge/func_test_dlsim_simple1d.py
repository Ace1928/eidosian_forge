import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_dlsim_simple1d(self):
    a = np.array([[0.5]])
    b = np.array([[0.0]])
    c = np.array([[1.0]])
    d = np.array([[0.0]])
    n = 5
    u = np.zeros(n).reshape(-1, 1)
    tout, yout, xout = dlsim((a, b, c, d, 1), u, x0=1)
    assert_array_equal(tout, np.arange(float(n)))
    expected = (0.5 ** np.arange(float(n))).reshape(-1, 1)
    assert_array_equal(yout, expected)
    assert_array_equal(xout, expected)