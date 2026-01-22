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
def test_sym_boundary(self):
    a = [[1, 2, 3], [3, 4, 5]]
    b = [[2, 3, 4], [4, 5, 6]]
    c = convolve2d(a, b, 'full', 'symm')
    d = array([[34, 30, 44, 62, 66], [52, 48, 62, 80, 84], [82, 78, 92, 110, 114]])
    assert_array_equal(c, d)