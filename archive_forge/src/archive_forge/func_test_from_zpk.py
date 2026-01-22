import numpy as np
from numpy.testing import (assert_equal,
from pytest import raises as assert_raises
from scipy.signal import (dlsim, dstep, dimpulse, tf2zpk, lti, dlti,
def test_from_zpk(self):
    system_ZPK = dlti([], [0.2], 0.3)
    system_TF = dlti(0.3, [1, -0.2])
    w = [0.1, 1, 10, 100]
    w1, H1 = dfreqresp(system_ZPK, w=w)
    w2, H2 = dfreqresp(system_TF, w=w)
    assert_almost_equal(H1, H2)