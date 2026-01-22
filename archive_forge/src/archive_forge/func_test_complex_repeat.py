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
def test_complex_repeat(self):
    p = [-1.0, -1.0 + 0.05j, -0.95 + 0.15j, -0.9 + 0.15j, 0.0, 0.5 + 0.5j, 0.45 + 0.55j]
    unique, multiplicity = unique_roots(p, tol=0.1, rtype='min')
    assert_almost_equal(unique, [-1.0, -0.95 + 0.15j, 0.0, 0.45 + 0.55j], decimal=15)
    assert_equal(multiplicity, [2, 2, 1, 2])
    unique, multiplicity = unique_roots(p, tol=0.1, rtype='max')
    assert_almost_equal(unique, [-1.0 + 0.05j, -0.9 + 0.15j, 0.0, 0.5 + 0.5j], decimal=15)
    assert_equal(multiplicity, [2, 2, 1, 2])
    unique, multiplicity = unique_roots(p, tol=0.1, rtype='avg')
    assert_almost_equal(unique, [-1.0 + 0.025j, -0.925 + 0.15j, 0.0, 0.475 + 0.525j], decimal=15)
    assert_equal(multiplicity, [2, 2, 1, 2])