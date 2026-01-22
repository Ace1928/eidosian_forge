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
def test_valid_mode2(self):
    e = [[1, 2, 3], [3, 4, 5]]
    f = [[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]]
    expected = [[62, 80, 98, 116, 134]]
    out = convolve2d(e, f, 'valid')
    assert_array_equal(out, expected)
    out = convolve2d(f, e, 'valid')
    assert_array_equal(out, expected)
    e = [[1 + 1j, 2 - 3j], [3 + 1j, 4 + 0j]]
    f = [[2 - 1j, 3 + 2j, 4 + 0j], [4 - 0j, 5 + 1j, 6 - 3j]]
    expected = [[27 - 1j, 46.0 + 2j]]
    out = convolve2d(e, f, 'valid')
    assert_array_equal(out, expected)
    out = convolve2d(f, e, 'valid')
    assert_array_equal(out, expected)