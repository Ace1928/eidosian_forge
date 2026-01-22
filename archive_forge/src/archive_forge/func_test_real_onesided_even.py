import sys
import numpy as np
from numpy.testing import (assert_, assert_approx_equal,
import pytest
from pytest import raises as assert_raises
from scipy import signal
from scipy.fft import fftfreq
from scipy.integrate import trapezoid
from scipy.signal import (periodogram, welch, lombscargle, coherence,
from scipy.signal._spectral_py import _spectral_helper
from scipy.signal.tests._scipy_spectral_test_shim import stft_compare as stft
from scipy.signal.tests._scipy_spectral_test_shim import istft_compare as istft
from scipy.signal.tests._scipy_spectral_test_shim import csd_compare as csd
def test_real_onesided_even(self):
    x = np.zeros(16)
    x[0] = 1
    x[8] = 1
    f, p = csd(x, x, nperseg=8)
    assert_allclose(f, np.linspace(0, 0.5, 5))
    q = np.array([0.08333333, 0.15277778, 0.22222222, 0.22222222, 0.11111111])
    assert_allclose(p, q, atol=1e-07, rtol=1e-07)