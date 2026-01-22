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
def test_permute_axes():
    """Verify correctness of four-dimensional signal by permuting its
    shape. """
    n = 25
    SFT = ShortTimeFFT(np.ones(8) / 8, hop=3, fs=n)
    x0 = np.arange(n)
    Sx0 = SFT.stft(x0)
    Sx0 = Sx0.reshape((Sx0.shape[0], 1, 1, 1, Sx0.shape[-1]))
    SxT = np.moveaxis(Sx0, (0, -1), (-1, 0))
    atol = 2 * np.finfo(SFT.win.dtype).resolution
    for i in range(4):
        y = np.reshape(x0, np.roll((n, 1, 1, 1), i))
        Sy = SFT.stft(y, axis=i)
        assert_allclose(Sy, np.moveaxis(Sx0, 0, i))
        yb0 = SFT.istft(Sy, k1=n, f_axis=i)
        assert_allclose(yb0, y, atol=atol)
        yb1 = SFT.istft(Sy, k1=n, f_axis=i, t_axis=Sy.ndim - 1)
        assert_allclose(yb1, y, atol=atol)
        SyT = np.moveaxis(Sy, (i, -1), (-1, i))
        assert_allclose(SyT, np.moveaxis(SxT, 0, i))
        ybT = SFT.istft(SyT, k1=n, t_axis=i, f_axis=-1)
        assert_allclose(ybT, y, atol=atol)