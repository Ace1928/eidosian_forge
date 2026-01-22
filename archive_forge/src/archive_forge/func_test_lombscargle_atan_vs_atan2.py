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
def test_lombscargle_atan_vs_atan2(self):
    t = np.linspace(0, 10, 1000, endpoint=False)
    x = np.sin(4 * t)
    f = np.linspace(0, 50, 500, endpoint=False) + 0.1
    lombscargle(t, x, f * 2 * np.pi)