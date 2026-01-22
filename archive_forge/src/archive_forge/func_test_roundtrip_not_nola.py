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
def test_roundtrip_not_nola(self):
    np.random.seed(1234)
    w_fail = np.ones(16)
    w_fail[::2] = 0
    settings = [(w_fail, 256, len(w_fail), len(w_fail) // 2), ('hann', 256, 64, 0)]
    for window, N, nperseg, noverlap in settings:
        msg = f'{window}, {N}, {nperseg}, {noverlap}'
        assert not check_NOLA(window, nperseg, noverlap), msg
        t = np.arange(N)
        x = 10 * np.random.randn(t.size)
        _, _, zz = stft(x, nperseg=nperseg, noverlap=noverlap, window=window, detrend=None, padded=True, boundary='zeros')
        with pytest.warns(UserWarning, match='NOLA'):
            tr, xr = istft(zz, nperseg=nperseg, noverlap=noverlap, window=window, boundary=True)
        assert np.allclose(t, tr[:len(t)]), msg
        assert not np.allclose(x, xr[:len(x)]), msg