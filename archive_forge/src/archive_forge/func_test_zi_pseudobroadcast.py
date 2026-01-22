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
def test_zi_pseudobroadcast(self):
    x = self.generate((4, 5, 20))
    b, a = signal.butter(8, 0.2, output='ba')
    b = self.convert_dtype(b)
    a = self.convert_dtype(a)
    zi_size = b.shape[0] - 1
    zi_full = self.convert_dtype(np.ones((4, 5, zi_size)))
    zi_sing = self.convert_dtype(np.ones((1, 1, zi_size)))
    y_full, zf_full = lfilter(b, a, x, zi=zi_full)
    y_sing, zf_sing = lfilter(b, a, x, zi=zi_sing)
    assert_array_almost_equal(y_sing, y_full)
    assert_array_almost_equal(zf_full, zf_sing)
    assert_raises(ValueError, lfilter, b, a, x, -1, np.ones(zi_size))