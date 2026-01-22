import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_linear_freq_01(self):
    method = 'linear'
    f0 = 1.0
    f1 = 2.0
    t1 = 1.0
    t = np.linspace(0, t1, 100)
    phase = waveforms._chirp_phase(t, f0, t1, f1, method)
    tf, f = compute_frequency(t, phase)
    abserr = np.max(np.abs(f - chirp_linear(tf, f0, f1, t1)))
    assert_(abserr < 1e-06)