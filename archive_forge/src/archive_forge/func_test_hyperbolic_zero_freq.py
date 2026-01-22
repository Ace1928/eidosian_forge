import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_hyperbolic_zero_freq(self):
    method = 'hyperbolic'
    t1 = 1.0
    t = np.linspace(0, t1, 5)
    assert_raises(ValueError, waveforms.chirp, t, 0, t1, 1, method)
    assert_raises(ValueError, waveforms.chirp, t, 1, t1, 0, method)