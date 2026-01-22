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
@pytest.mark.parametrize('fft_mode, f', [('onesided', [0.0, 1.0, 2.0]), ('onesided2X', [0.0, 1.0, 2.0]), ('twosided', [0.0, 1.0, 2.0, -2.0, -1.0]), ('centered', [-2.0, -1.0, 0.0, 1.0, 2.0])])
def test_f(fft_mode: FFT_MODE_TYPE, f):
    """Verify the frequency values property `f`."""
    SFT = ShortTimeFFT(np.ones(5), hop=4, fs=5, fft_mode=fft_mode, scale_to='psd')
    assert_equal(SFT.f, f)