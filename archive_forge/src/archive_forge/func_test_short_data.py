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
def test_short_data(self):
    x = np.random.randn(1024)
    fs = 1.0
    f, _, p = spectrogram(x, fs, window=('tukey', 0.25))
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'nperseg = 1025 is greater than input length  = 1024, using nperseg = 1024')
        f1, _, p1 = spectrogram(x, fs, window=('tukey', 0.25), nperseg=1025)
    f2, _, p2 = spectrogram(x, fs, nperseg=256)
    f3, _, p3 = spectrogram(x, fs, nperseg=1024)
    assert_allclose(f, f2)
    assert_allclose(p, p2)
    assert_allclose(f1, f3)
    assert_allclose(p1, p3)