import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_dlti_instantiation(self):
    dt = 0.05
    s = dlti([1], [-1], dt=dt)
    assert_(isinstance(s, TransferFunction))
    assert_(isinstance(s, dlti))
    assert_(not isinstance(s, lti))
    assert_equal(s.dt, dt)
    s = dlti(np.array([]), np.array([-1]), 1, dt=dt)
    assert_(isinstance(s, ZerosPolesGain))
    assert_(isinstance(s, dlti))
    assert_(not isinstance(s, lti))
    assert_equal(s.dt, dt)
    s = dlti([1], [-1], 1, 3, dt=dt)
    assert_(isinstance(s, StateSpace))
    assert_(isinstance(s, dlti))
    assert_(not isinstance(s, lti))
    assert_equal(s.dt, dt)
    assert_raises(ValueError, dlti, 1)
    assert_raises(ValueError, dlti, 1, 1, 1, 1, 1)