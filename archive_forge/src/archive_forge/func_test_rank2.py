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
def test_rank2(self, dt):
    shape = (4, 3)
    x = np.linspace(0, np.prod(shape) - 1, np.prod(shape)).reshape(shape)
    x = x.astype(dt)
    b = np.array([1, -1]).astype(dt)
    a = np.array([0.5, 0.5]).astype(dt)
    y_r2_a0 = np.array([[0, 2, 4], [6, 4, 2], [0, 2, 4], [6, 4, 2]], dtype=dt)
    y_r2_a1 = np.array([[0, 2, 0], [6, -4, 6], [12, -10, 12], [18, -16, 18]], dtype=dt)
    y = sosfilt(tf2sos(b, a), x, axis=0)
    assert_array_almost_equal(y_r2_a0, y)
    y = sosfilt(tf2sos(b, a), x, axis=1)
    assert_array_almost_equal(y_r2_a1, y)