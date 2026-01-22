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
def test_invres(self):
    b, a = invres([1], [1], [])
    assert_almost_equal(b, [1])
    assert_almost_equal(a, [1, -1])
    b, a = invres([1 - 1j, 2, 0.5 - 3j], [1, 0.5j, 1 + 1j], [])
    assert_almost_equal(b, [3.5 - 4j, -8.5 + 0.25j, 3.5 + 3.25j])
    assert_almost_equal(a, [1, -2 - 1.5j, 0.5 + 2j, 0.5 - 0.5j])
    b, a = invres([0.5, 1], [1 - 1j, 2 + 2j], [1, 2, 3])
    assert_almost_equal(b, [1, -1 - 1j, 1 - 2j, 0.5 - 3j, 10])
    assert_almost_equal(a, [1, -3 - 1j, 4])
    b, a = invres([-1, 2, 1j, 3 - 1j, 4, -2], [-1, 2 - 1j, 2 - 1j, 3, 3, 3], [])
    assert_almost_equal(b, [4 - 1j, -28 + 16j, 40 - 62j, 100 + 24j, -292 + 219j, 192 - 268j])
    assert_almost_equal(a, [1, -12 + 2j, 53 - 20j, -96 + 68j, 27 - 72j, 108 - 54j, -81 + 108j])
    b, a = invres([-1, 1j], [1, 1], [1, 2])
    assert_almost_equal(b, [1, 0, -4, 3 + 1j])
    assert_almost_equal(a, [1, -2, 1])