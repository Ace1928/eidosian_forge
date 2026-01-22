import numpy as np
from numpy.testing import \
import pytest
from scipy.signal import cont2discrete as c2d
from scipy.signal import dlsim, ss2tf, ss2zpk, lsim, lti
from scipy.signal import tf2ss, impulse, dimpulse, step, dstep
def test_gbt_with_sio_tf_and_zpk(self):
    """Test method='gbt' with alpha=0.25 for tf and zpk cases."""
    A = -1.0
    B = 1.0
    C = 1.0
    D = 0.5
    cnum, cden = ss2tf(A, B, C, D)
    cz, cp, ck = ss2zpk(A, B, C, D)
    h = 1.0
    alpha = 0.25
    Ad = (1 + (1 - alpha) * h * A) / (1 - alpha * h * A)
    Bd = h * B / (1 - alpha * h * A)
    Cd = C / (1 - alpha * h * A)
    Dd = D + alpha * C * Bd
    dnum, dden = ss2tf(Ad, Bd, Cd, Dd)
    c2dnum, c2dden, dt = c2d((cnum, cden), h, method='gbt', alpha=alpha)
    assert_allclose(dnum, c2dnum)
    assert_allclose(dden, c2dden)
    dz, dp, dk = ss2zpk(Ad, Bd, Cd, Dd)
    c2dz, c2dp, c2dk, dt = c2d((cz, cp, ck), h, method='gbt', alpha=alpha)
    assert_allclose(dz, c2dz)
    assert_allclose(dp, c2dp)
    assert_allclose(dk, c2dk)