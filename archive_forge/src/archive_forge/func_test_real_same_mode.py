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
@pytest.mark.parametrize('axes', ['', None, 0, [0], -1, [-1]])
def test_real_same_mode(self, axes):
    a = array([1, 2, 3])
    b = array([3, 3, 5, 6, 8, 7, 9, 0, 1])
    expected_1 = array([35.0, 41.0, 47.0])
    expected_2 = array([9.0, 20.0, 25.0, 35.0, 41.0, 47.0, 39.0, 28.0, 2.0])
    if axes == '':
        out = fftconvolve(a, b, 'same')
    else:
        out = fftconvolve(a, b, 'same', axes=axes)
    assert_array_almost_equal(out, expected_1)
    if axes == '':
        out = fftconvolve(b, a, 'same')
    else:
        out = fftconvolve(b, a, 'same', axes=axes)
    assert_array_almost_equal(out, expected_2)