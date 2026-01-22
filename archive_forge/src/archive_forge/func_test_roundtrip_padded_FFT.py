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
def test_roundtrip_padded_FFT(self):
    np.random.seed(1234)
    settings = [('hann', 1024, 256, 128, 512), ('hann', 1024, 256, 128, 501), ('boxcar', 100, 10, 0, 33), (('tukey', 0.5), 1152, 256, 64, 1024)]
    for window, N, nperseg, noverlap, nfft in settings:
        t = np.arange(N)
        x = 10 * np.random.randn(t.size)
        xc = x * np.exp(1j * np.pi / 4)
        _, _, z = stft(x, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, detrend=None, padded=True)
        _, _, zc = stft(xc, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, detrend=None, padded=True, return_onesided=False)
        tr, xr = istft(z, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window)
        tr, xcr = istft(zc, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window, input_onesided=False)
        msg = f'{window}, {noverlap}'
        assert_allclose(t, tr, err_msg=msg)
        assert_allclose(x, xr, err_msg=msg)
        assert_allclose(xc, xcr, err_msg=msg)