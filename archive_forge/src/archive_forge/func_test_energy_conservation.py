import math
from itertools import product
from typing import cast, get_args, Literal
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from scipy.fft import fftshift
from scipy.stats import norm as normal_distribution  # type: ignore
from scipy.signal import get_window, welch, stft, istft, spectrogram
from scipy.signal._short_time_fft import FFT_MODE_TYPE, \
from scipy.signal.windows import gaussian
@pytest.mark.parametrize('N_x', (128, 129, 255, 256, 1337))
@pytest.mark.parametrize('w_size', (128, 256))
@pytest.mark.parametrize('t_step', (4, 64))
@pytest.mark.parametrize('f_c', (7.0, 23.0))
def test_energy_conservation(N_x: int, w_size: int, t_step: int, f_c: float):
    """Test if a `psd`-scaled STFT conserves the L2 norm.

    This test is adapted from MNE-Python [1]_. Besides being battle-tested,
    this test has the benefit of using non-standard window including
    non-positive values and a 2d input signal.

    Since `ShortTimeFFT` requires the signal length `N_x` to be at least the
    window length `w_size`, the parameter `N_x` was changed from
    ``(127, 128, 255, 256, 1337)`` to ``(128, 129, 255, 256, 1337)`` to be
    more useful.

    .. [1] File ``test_stft.py`` of MNE-Python
        https://github.com/mne-tools/mne-python/blob/main/mne/time_frequency/tests/test_stft.py
    """
    window = np.sin(np.arange(0.5, w_size + 0.5) / w_size * np.pi)
    SFT = ShortTimeFFT(window, t_step, fs=1000, fft_mode='onesided2X', scale_to='psd')
    atol = 2 * np.finfo(window.dtype).resolution
    N_x = max(N_x, w_size)
    t = np.arange(N_x).astype(np.float64)
    x = np.sin(2 * np.pi * f_c * t * SFT.T)
    x = np.array([x, x + 1.0])
    X = SFT.stft(x)
    xp = SFT.istft(X, k1=N_x)
    max_freq = SFT.f[np.argmax(np.sum(np.abs(X[0]) ** 2, axis=1))]
    assert X.shape[1] == SFT.f_pts
    assert np.all(SFT.f >= 0.0)
    assert np.abs(max_freq - f_c) < 1.0
    assert_allclose(x, xp, atol=atol)
    E_x = np.sum(x ** 2, axis=-1) * SFT.T
    aX2 = X.real ** 2 + X.imag.real ** 2
    E_X = np.sum(np.sum(aX2, axis=-1) * SFT.delta_t, axis=-1) * SFT.delta_f
    assert_allclose(E_X, E_x, atol=atol)
    np.random.seed(2392795)
    x = np.random.randn(2, N_x)
    X = SFT.stft(x)
    xp = SFT.istft(X, k1=N_x)
    assert X.shape[1] == SFT.f_pts
    assert np.all(SFT.f >= 0.0)
    assert np.abs(max_freq - f_c) < 1.0
    assert_allclose(x, xp, atol=atol)
    E_x = np.sum(x ** 2, axis=-1) * SFT.T
    aX2 = X.real ** 2 + X.imag.real ** 2
    E_X = np.sum(np.sum(aX2, axis=-1) * SFT.delta_t, axis=-1) * SFT.delta_f
    assert_allclose(E_X, E_x, atol=atol)
    x = np.zeros((0, N_x))
    X = SFT.stft(x)
    xp = SFT.istft(X, k1=N_x)
    assert xp.shape == x.shape