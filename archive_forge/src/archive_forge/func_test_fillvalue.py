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
def test_fillvalue(self):
    a = [[1, 2, 3], [3, 4, 5]]
    b = [[2, 3, 4], [4, 5, 6]]
    fillval = 1
    c = convolve2d(a, b, 'full', 'fill', fillval)
    d = array([[24, 26, 31, 34, 32], [28, 40, 62, 64, 52], [32, 46, 67, 62, 48]])
    assert_array_equal(c, d)