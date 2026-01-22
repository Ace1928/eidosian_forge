import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_boxcar(self):
    w = windows.get_window('boxcar', 12)
    assert_array_equal(w, np.ones_like(w))
    w = windows.get_window(('boxcar',), 16)
    assert_array_equal(w, np.ones_like(w))