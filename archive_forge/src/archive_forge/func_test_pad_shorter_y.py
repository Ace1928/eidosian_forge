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
def test_pad_shorter_y(self):
    x = np.zeros(12)
    y = np.zeros(8)
    f = np.linspace(0, 0.5, 7)
    c = np.zeros(7, dtype=np.complex128)
    f1, c1 = csd(x, y, nperseg=12)
    assert_allclose(f, f1)
    assert_allclose(c, c1)