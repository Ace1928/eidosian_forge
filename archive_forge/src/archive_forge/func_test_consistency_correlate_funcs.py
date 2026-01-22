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
def test_consistency_correlate_funcs(self):
    a = np.arange(5)
    b = np.array([3.2, 1.4, 3])
    for mode in ['full', 'valid', 'same']:
        assert_almost_equal(np.correlate(a, b, mode=mode), signal.correlate(a, b, mode=mode))
        assert_almost_equal(np.squeeze(signal.correlate2d([a], [b], mode=mode)), signal.correlate(a, b, mode=mode))
        if mode == 'valid':
            assert_almost_equal(np.correlate(b, a, mode=mode), signal.correlate(b, a, mode=mode))
            assert_almost_equal(np.squeeze(signal.correlate2d([b], [a], mode=mode)), signal.correlate(b, a, mode=mode))