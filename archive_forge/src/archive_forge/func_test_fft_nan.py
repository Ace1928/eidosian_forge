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
def test_fft_nan(self):
    n = 1000
    rng = np.random.default_rng(43876432987)
    sig_nan = rng.standard_normal(n)
    for val in [np.nan, np.inf]:
        sig_nan[100] = val
        coeffs = signal.firwin(200, 0.2)
        msg = 'Use of fft convolution.*|invalid value encountered.*'
        with pytest.warns(RuntimeWarning, match=msg):
            signal.convolve(sig_nan, coeffs, mode='same', method='fft')