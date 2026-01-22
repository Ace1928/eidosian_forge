import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_array_as_window(self):
    osfactor = 128
    sig = np.arange(128)
    win = windows.get_window(('kaiser', 8.0), osfactor // 2)
    with assert_raises(ValueError, match='must have the same length'):
        resample(sig, len(sig) * osfactor, window=win)