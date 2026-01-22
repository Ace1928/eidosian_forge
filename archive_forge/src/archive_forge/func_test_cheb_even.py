import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_cheb_even(self):
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'This window is not suitable')
        w = windows.get_window(('chebwin', 40), 54, fftbins=False)
    assert_array_almost_equal(w, cheb_even_true, decimal=4)