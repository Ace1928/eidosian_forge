import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def test_logarithmic_at_zero(self):
    w = waveforms.chirp(t=0, f0=1.0, f1=2.0, t1=1.0, method='logarithmic')
    assert_almost_equal(w, 1.0)