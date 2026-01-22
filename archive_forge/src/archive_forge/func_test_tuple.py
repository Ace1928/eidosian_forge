import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def test_tuple(self):
    taps1 = firwin2(150, (0.0, 0.5, 0.5, 1.0), (1.0, 1.0, 0.0, 0.0))
    taps2 = firwin2(150, [0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 0.0, 0.0])
    assert_array_almost_equal(taps1, taps2)