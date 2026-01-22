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
def test_rank_1_IIR(self):
    x = self.generate((6,))
    b = self.convert_dtype([1, -1])
    a = self.convert_dtype([0.5, -0.5])
    y_r = self.convert_dtype([0, 2, 4, 6, 8, 10.0])
    assert_array_almost_equal(lfilter(b, a, x), y_r)