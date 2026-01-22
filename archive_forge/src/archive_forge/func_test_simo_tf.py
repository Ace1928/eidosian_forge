import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
def test_simo_tf(self):
    tf = ([[1, 0], [1, 1]], [1, 1])
    num, den, dt = c2d(tf, 0.01)
    assert_equal(dt, 0.01)
    assert_allclose(den, [1, -0.990404983], rtol=0.001)
    assert_allclose(num, [[1, -1], [1, -0.99004983]], rtol=0.001)