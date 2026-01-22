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
def test_valid_mode_ignore_nonaxes(self):
    a = array([3, 2, 1])
    b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
    expected = array([24.0, 31.0, 41.0, 43.0, 49.0, 25.0, 12.0])
    a = np.tile(a, [2, 1])
    b = np.tile(b, [1, 1])
    expected = np.tile(expected, [2, 1])
    out = fftconvolve(a, b, 'valid', axes=1)
    assert_array_almost_equal(out, expected)