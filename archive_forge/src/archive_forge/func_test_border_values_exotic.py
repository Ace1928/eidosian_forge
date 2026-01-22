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
def test_border_values_exotic():
    """Ensure that the border calculations are correct for windows with
    zeros. """
    w = np.array([0, 0, 0, 0, 0, 0, 0, 1.0])
    SFT = ShortTimeFFT(w, hop=1, fs=1)
    assert SFT.lower_border_end == (0, 0)
    SFT = ShortTimeFFT(np.flip(w), hop=20, fs=1)
    assert SFT.upper_border_begin(4) == (0, 0)
    SFT._hop = -1
    with pytest.raises(RuntimeError):
        _ = SFT.k_max(4)
    with pytest.raises(RuntimeError):
        _ = SFT.k_min