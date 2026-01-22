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
@pytest.mark.parametrize('signal_type', ('real', 'complex'))
def test_roundtrip_complex_window(signal_type):
    """Test roundtrip for complex-valued window function

    The purpose of this test is to check if the dual window is calculated
    correctly for complex-valued windows.
    """
    np.random.seed(1354654)
    win = np.exp(2j * np.linspace(0, np.pi, 8))
    SFT = ShortTimeFFT(win, 3, fs=1, fft_mode='twosided')
    z = 10 * np.random.randn(11)
    if signal_type == 'complex':
        z = z + 2j * z
    Sz = SFT.stft(z)
    z1 = SFT.istft(Sz, k1=len(z))
    assert_allclose(z, z1, err_msg='Roundtrip for complex-valued window failed')