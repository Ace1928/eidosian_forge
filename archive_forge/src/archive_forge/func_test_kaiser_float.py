import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_kaiser_float(self):
    win1 = windows.get_window(7.2, 64)
    win2 = windows.kaiser(64, 7.2, False)
    assert_allclose(win1, win2)