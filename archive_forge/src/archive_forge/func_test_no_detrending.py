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
def test_no_detrending(self):
    x = np.arange(10, dtype=np.float64) + 0.04
    f1, p1 = csd(x, x, nperseg=10, detrend=False)
    f2, p2 = csd(x, x, nperseg=10, detrend=lambda x: x)
    assert_allclose(f1, f2, atol=1e-15)
    assert_allclose(p1, p2, atol=1e-15)