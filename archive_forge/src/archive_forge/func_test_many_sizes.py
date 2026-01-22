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
@pytest.mark.slow
@pytest.mark.parametrize('n', list(range(1, 100)) + list(range(1000, 1500)) + np.random.RandomState(1234).randint(1001, 10000, 5).tolist())
def test_many_sizes(self, n):
    a = np.random.rand(n) + 1j * np.random.rand(n)
    b = np.random.rand(n) + 1j * np.random.rand(n)
    expected = np.convolve(a, b, 'full')
    out = fftconvolve(a, b, 'full')
    assert_allclose(out, expected, atol=1e-10)
    out = fftconvolve(a, b, 'full', axes=[0])
    assert_allclose(out, expected, atol=1e-10)