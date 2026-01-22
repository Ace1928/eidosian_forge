import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_pole_one(self):
    system = TransferFunction([1], [1, -1], dt=0.1)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, message='divide by zero')
        sup.filter(RuntimeWarning, message='invalid value encountered')
        w, mag, phase = dbode(system, n=2)
    assert_equal(w[0], 0.0)