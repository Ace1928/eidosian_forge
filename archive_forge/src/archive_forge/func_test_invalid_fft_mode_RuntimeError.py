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
def test_invalid_fft_mode_RuntimeError():
    """Ensure exception gets raised when property `fft_mode` is invalid. """
    SFT = ShortTimeFFT(np.ones(8), hop=4, fs=1)
    SFT._fft_mode = 'invalid_typ'
    with pytest.raises(RuntimeError):
        _ = SFT.f
    with pytest.raises(RuntimeError):
        SFT._fft_func(np.ones(8))
    with pytest.raises(RuntimeError):
        SFT._ifft_func(np.ones(8))