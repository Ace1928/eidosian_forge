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
def test_zi_some_singleton_dims(self):
    x = self.convert_dtype(np.zeros((3, 2, 5), 'l'))
    b = self.convert_dtype(np.ones(5, 'l'))
    a = self.convert_dtype(np.array([1, 0, 0]))
    zi = np.ones((3, 1, 4), 'l')
    zi[1, :, :] *= 2
    zi[2, :, :] *= 3
    zi = self.convert_dtype(zi)
    zf_expected = self.convert_dtype(np.zeros((3, 2, 4), 'l'))
    y_expected = np.zeros((3, 2, 5), 'l')
    y_expected[:, :, :4] = [[[1]], [[2]], [[3]]]
    y_expected = self.convert_dtype(y_expected)
    y_iir, zf_iir = lfilter(b, a, x, -1, zi)
    assert_array_almost_equal(y_iir, y_expected)
    assert_array_almost_equal(zf_iir, zf_expected)
    y_fir, zf_fir = lfilter(b, a[0], x, -1, zi)
    assert_array_almost_equal(y_fir, y_expected)
    assert_array_almost_equal(zf_fir, zf_expected)