import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def chirp_hyperbolic(t, f0, f1, t1):
    f = f0 * f1 * t1 / ((f0 - f1) * t + f1 * t1)
    return f