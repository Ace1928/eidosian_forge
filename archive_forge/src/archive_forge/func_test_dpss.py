import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_dpss(self):
    win1 = windows.get_window(('dpss', 3), 64, fftbins=False)
    win2 = windows.dpss(64, 3)
    assert_array_almost_equal(win1, win2, decimal=4)