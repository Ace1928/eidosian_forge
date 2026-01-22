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
def filtfilt_gust_opt_func(ics, b, a, x):
    """Objective function used in filtfilt_gust_opt."""
    m = max(len(a), len(b)) - 1
    z0f = ics[:m]
    z0b = ics[m:]
    y_f = lfilter(b, a, x, zi=z0f)[0]
    y_fb = lfilter(b, a, y_f[::-1], zi=z0b)[0][::-1]
    y_b = lfilter(b, a, x[::-1], zi=z0b)[0][::-1]
    y_bf = lfilter(b, a, y_b, zi=z0f)[0]
    value = np.sum((y_fb - y_bf) ** 2)
    return value