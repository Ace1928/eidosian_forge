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
def test_nd_axis_0(self):
    x = np.arange(20, dtype=np.float64) + 0.04
    x = x.reshape((10, 2, 1))
    f, p = csd(x, x, nperseg=10, axis=0)
    assert_array_equal(p.shape, (6, 2, 1))
    assert_allclose(p[:, 0, 0], p[:, 1, 0], atol=1e-13, rtol=1e-13)
    f0, p0 = csd(x[:, 0, 0], x[:, 0, 0], nperseg=10)
    assert_allclose(p0, p[:, 1, 0], atol=1e-13, rtol=1e-13)