import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal,
from pytest import raises as assert_raises
import scipy.signal._waveforms as waveforms
def compute_frequency(t, theta):
    """
    Compute theta'(t)/(2*pi), where theta'(t) is the derivative of theta(t).
    """
    dt = t[1] - t[0]
    f = np.diff(theta) / (2 * np.pi) / dt
    tf = 0.5 * (t[1:] + t[:-1])
    return (tf, f)