import pickle
import numpy as np
from numpy import array
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.signal import windows, get_window, resample, hann as dep_hann
from scipy import signal
def test_not_needs_params():
    for winstr in ['barthann', 'bartlett', 'blackman', 'blackmanharris', 'bohman', 'boxcar', 'cosine', 'flattop', 'hamming', 'nuttall', 'parzen', 'taylor', 'exponential', 'poisson', 'tukey', 'tuk', 'triangle', 'lanczos', 'sinc']:
        win = get_window(winstr, 7)
        assert_equal(len(win), 7)