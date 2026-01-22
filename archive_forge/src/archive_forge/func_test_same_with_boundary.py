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
@pytest.mark.parametrize('func', [convolve2d, correlate2d])
@pytest.mark.parametrize('boundary, expected', [('symm', [[37.0, 42.0, 44.0, 45.0]]), ('wrap', [[43.0, 44.0, 42.0, 39.0]])])
def test_same_with_boundary(self, func, boundary, expected):
    image = np.array([[2.0, -1.0, 3.0, 4.0]])
    kernel = np.ones((1, 21))
    result = func(image, kernel, mode='same', boundary=boundary)
    assert_array_equal(result, expected)