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
def test_frequency(self):
    """Test if frequency location of peak corresponds to frequency of
        generated input signal.
        """
    ampl = 2.0
    w = 1.0
    phi = 0.5 * np.pi
    nin = 100
    nout = 1000
    p = 0.7
    np.random.seed(2353425)
    r = np.random.rand(nin)
    t = np.linspace(0.01 * np.pi, 10.0 * np.pi, nin)[r >= p]
    x = ampl * np.sin(w * t + phi)
    f = np.linspace(0.01, 10.0, nout)
    P = lombscargle(t, x, f)
    delta = f[1] - f[0]
    assert_(w - f[np.argmax(P)] < delta / 2.0)