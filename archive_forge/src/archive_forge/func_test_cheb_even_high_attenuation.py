import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_cheb_even_high_attenuation(self):
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'This window is not suitable')
        cheb_even = windows.chebwin(54, at=40)
    assert_array_almost_equal(cheb_even, cheb_even_true, decimal=4)