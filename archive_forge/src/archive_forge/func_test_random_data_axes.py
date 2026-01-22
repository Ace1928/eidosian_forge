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
@pytest.mark.parametrize('axes', [1, [1], -1, [-1]])
def test_random_data_axes(self, axes):
    np.random.seed(1234)
    a = np.random.rand(1233) + 1j * np.random.rand(1233)
    b = np.random.rand(1321) + 1j * np.random.rand(1321)
    expected = np.convolve(a, b, 'full')
    a = np.tile(a, [2, 1])
    b = np.tile(b, [2, 1])
    expected = np.tile(expected, [2, 1])
    out = fftconvolve(a, b, 'full', axes=axes)
    assert_(np.allclose(out, expected, rtol=1e-10))