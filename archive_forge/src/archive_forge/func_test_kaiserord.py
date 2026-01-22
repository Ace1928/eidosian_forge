import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_kaiserord():
    assert_raises(ValueError, kaiserord, 1.0, 1.0)
    numtaps, beta = kaiserord(2.285 + 7.95 - 0.001, 1 / np.pi)
    assert_equal((numtaps, beta), (2, 0.0))