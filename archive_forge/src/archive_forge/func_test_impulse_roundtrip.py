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
@pytest.mark.parametrize('i', range(19))
def test_impulse_roundtrip(i):
    """Roundtrip for an impulse being at different positions `i`."""
    n = 19
    w, h_n = (np.ones(8), 3)
    x = np.zeros(n)
    x[i] = 1
    SFT = ShortTimeFFT(w, hop=h_n, fs=1, scale_to=None, phase_shift=None)
    Sx = SFT.stft(x)
    n_q = SFT.nearest_k_p(n // 2)
    Sx0 = SFT.stft(x[:n_q], padding='zeros')
    Sx1 = SFT.stft(x[n_q:], padding='zeros')
    q0_ub = SFT.upper_border_begin(n_q)[1] - SFT.p_min
    q1_le = SFT.lower_border_end[1] - SFT.p_min
    assert_allclose(Sx0[:, :q0_ub], Sx[:, :q0_ub], err_msg=f'i={i!r}')
    assert_allclose(Sx1[:, q1_le:], Sx[:, q1_le - Sx1.shape[1]:], err_msg=f'i={i!r}')
    Sx01 = np.hstack((Sx0[:, :q0_ub], Sx0[:, q0_ub:] + Sx1[:, :q1_le], Sx1[:, q1_le:]))
    assert_allclose(Sx, Sx01, atol=1e-08, err_msg=f'i={i!r}')
    y = SFT.istft(Sx, 0, n)
    assert_allclose(y, x, atol=1e-08, err_msg=f'i={i!r}')
    y0 = SFT.istft(Sx, 0, n // 2)
    assert_allclose(x[:n // 2], y0, atol=1e-08, err_msg=f'i={i!r}')
    y1 = SFT.istft(Sx, n // 2, n)
    assert_allclose(x[n // 2:], y1, atol=1e-08, err_msg=f'i={i!r}')