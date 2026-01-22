import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_general_hamming(self):
    assert_allclose(get_window(('general_hamming', 0.7), 5), [0.4, 0.6072949, 0.9427051, 0.9427051, 0.6072949])
    assert_allclose(get_window(('general_hamming', 0.7), 5, fftbins=False), [0.4, 0.7, 1.0, 0.7, 0.4])