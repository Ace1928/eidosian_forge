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
def test_detrend_external_nd_0(self):
    x = np.arange(20, dtype=np.float64) + 0.04
    x = x.reshape((2, 1, 10))
    x = np.moveaxis(x, 2, 0)
    f, p = csd(x, x, nperseg=10, axis=0, detrend=lambda seg: signal.detrend(seg, axis=0, type='l'))
    assert_allclose(p, np.zeros_like(p), atol=1e-15)