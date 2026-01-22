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
def test_residuez_general(self):
    r, p, k = residuez([1, 6, 6, 2], [1, -(2 + 1j), 1 + 2j, -1j])
    self.assert_rp_almost_equal(r, p, [-2 + 2.5j, 7.5 + 7.5j, -4.5 - 12j], [1j, 1, 1])
    assert_almost_equal(k, [2j])
    r, p, k = residuez([1, 2, 1], [1, -1, 0.3561])
    self.assert_rp_almost_equal(r, p, [-0.9041 - 5.9928j, -0.9041 + 5.9928j], [0.5 + 0.3257j, 0.5 - 0.3257j], decimal=4)
    assert_almost_equal(k, [2.8082], decimal=4)
    r, p, k = residuez([1, -1], [1, -5, 6])
    assert_almost_equal(r, [-1, 2])
    assert_almost_equal(p, [2, 3])
    assert_equal(k.size, 0)
    r, p, k = residuez([2, 3, 4], [1, 3, 3, 1])
    self.assert_rp_almost_equal(r, p, [4, -5, 3], [-1, -1, -1])
    assert_equal(k.size, 0)
    r, p, k = residuez([1, -10, -4, 4], [2, -2, -4])
    assert_almost_equal(r, [0.5, -1.5])
    assert_almost_equal(p, [-1, 2])
    assert_almost_equal(k, [1.5, -1])
    r, p, k = residuez([18], [18, 3, -4, -1])
    self.assert_rp_almost_equal(r, p, [0.36, 0.24, 0.4], [0.5, -1 / 3, -1 / 3])
    assert_equal(k.size, 0)
    r, p, k = residuez([2, 3], np.polymul([1, -1 / 2], [1, 1 / 4]))
    assert_almost_equal(r, [-10 / 3, 16 / 3])
    assert_almost_equal(p, [-0.25, 0.5])
    assert_equal(k.size, 0)
    r, p, k = residuez([1, -2, 1], [1, -1])
    assert_almost_equal(r, [0])
    assert_almost_equal(p, [1])
    assert_almost_equal(k, [1, -1])
    r, p, k = residuez(1, [1, -1j])
    assert_almost_equal(r, [1])
    assert_almost_equal(p, [1j])
    assert_equal(k.size, 0)
    r, p, k = residuez(1, [1, -1, 0.25])
    assert_almost_equal(r, [0, 1])
    assert_almost_equal(p, [0.5, 0.5])
    assert_equal(k.size, 0)
    r, p, k = residuez(1, [1, -0.75, 0.125])
    assert_almost_equal(r, [-1, 2])
    assert_almost_equal(p, [0.25, 0.5])
    assert_equal(k.size, 0)
    r, p, k = residuez([1, 6, 2], [1, -2, 1])
    assert_almost_equal(r, [-10, 9])
    assert_almost_equal(p, [1, 1])
    assert_almost_equal(k, [2])
    r, p, k = residuez([6, 2], [1, -2, 1])
    assert_almost_equal(r, [-2, 8])
    assert_almost_equal(p, [1, 1])
    assert_equal(k.size, 0)
    r, p, k = residuez([1, 6, 6, 2], [1, -2, 1])
    assert_almost_equal(r, [-24, 15])
    assert_almost_equal(p, [1, 1])
    assert_almost_equal(k, [10, 2])
    r, p, k = residuez([1, 0, 1], [1, 0, 0, 0, 0, -1])
    self.assert_rp_almost_equal(r, p, [0.2618 + 0.1902j, 0.2618 - 0.1902j, 0.4, 0.0382 - 0.1176j, 0.0382 + 0.1176j], [-0.809 + 0.5878j, -0.809 - 0.5878j, 1.0, 0.309 + 0.9511j, 0.309 - 0.9511j], decimal=4)
    assert_equal(k.size, 0)