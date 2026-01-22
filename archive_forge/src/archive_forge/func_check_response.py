import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
from pytest import raises as assert_raises
import pytest
from scipy.fft import fft
from scipy.special import sinc
from scipy.signal import kaiser_beta, kaiser_atten, kaiserord, \
def check_response(self, h, expected_response, tol=0.05):
    N = len(h)
    alpha = 0.5 * (N - 1)
    m = np.arange(0, N) - alpha
    for freq, expected in expected_response:
        actual = abs(np.sum(h * np.exp(-1j * np.pi * m * freq)))
        mse = abs(actual - expected) ** 2
        assert_(mse < tol, f'response not as expected, mse={mse:g} > {tol:g}')