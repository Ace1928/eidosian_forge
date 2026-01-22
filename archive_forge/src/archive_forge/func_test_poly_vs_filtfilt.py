import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from itertools import product
from math import gcd
import pytest
from pytest import raises as assert_raises
from numpy.testing import (
from numpy import array, arange
import numpy as np
from scipy.fft import fft
from scipy.ndimage import correlate1d
from scipy.optimize import fmin, linear_sum_assignment
from scipy import signal
from scipy.signal import (
from scipy.signal.windows import hann
from scipy.signal._signaltools import (_filtfilt_gust, _compute_factors,
from scipy.signal._upfirdn import _upfirdn_modes
from scipy._lib import _testutils
from scipy._lib._util import ComplexWarning, np_long, np_ulong
def test_poly_vs_filtfilt(self):
    random_state = np.random.RandomState(17)
    try_types = (int, np.float32, np.complex64, float, complex)
    size = 10000
    down_factors = [2, 11, 79]
    for dtype in try_types:
        x = random_state.randn(size).astype(dtype)
        if dtype in (np.complex64, np.complex128):
            x += 1j * random_state.randn(size)
        x[0] = 0
        x[-1] = 0
        for down in down_factors:
            h = signal.firwin(31, 1.0 / down, window='hamming')
            yf = filtfilt(h, 1.0, x, padtype='constant')[::down]
            hc = convolve(h, h[::-1])
            y = signal.resample_poly(x, 1, down, window=hc)
            assert_allclose(yf, y, atol=1e-07, rtol=1e-07)