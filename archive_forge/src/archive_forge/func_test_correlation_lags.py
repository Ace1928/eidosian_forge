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
@pytest.mark.parametrize('mode', ['valid', 'same', 'full'])
@pytest.mark.parametrize('behind', [True, False])
@pytest.mark.parametrize('input_size', [100, 101, 1000, 1001, 10000, 10001])
def test_correlation_lags(mode, behind, input_size):
    rng = np.random.RandomState(0)
    in1 = rng.standard_normal(input_size)
    offset = int(input_size / 10)
    if behind:
        in2 = np.concatenate([rng.standard_normal(offset), in1])
        expected = -offset
    else:
        in2 = in1[offset:]
        expected = offset
    correlation = correlate(in1, in2, mode=mode)
    lags = correlation_lags(in1.size, in2.size, mode=mode)
    lag_index = np.argmax(correlation)
    assert_equal(lags[lag_index], expected)
    assert_equal(lags.shape, correlation.shape)