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
def test_axis_rolling(self):
    np.random.seed(1234)
    x_flat = np.random.randn(1024)
    _, _, z_flat = stft(x_flat)
    for a in range(3):
        newshape = [1] * 3
        newshape[a] = -1
        x = x_flat.reshape(newshape)
        _, _, z_plus = stft(x, axis=a)
        _, _, z_minus = stft(x, axis=a - x.ndim)
        assert_equal(z_flat, z_plus.squeeze(), err_msg=a)
        assert_equal(z_flat, z_minus.squeeze(), err_msg=a - x.ndim)
    _, x_transpose_m = istft(z_flat.T, time_axis=-2, freq_axis=-1)
    _, x_transpose_p = istft(z_flat.T, time_axis=0, freq_axis=1)
    assert_allclose(x_flat, x_transpose_m, err_msg='istft transpose minus')
    assert_allclose(x_flat, x_transpose_p, err_msg='istft transpose plus')