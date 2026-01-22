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
def test_fillvalue_errors(self):
    msg = 'could not cast `fillvalue` directly to the output '
    with np.testing.suppress_warnings() as sup:
        sup.filter(ComplexWarning, 'Casting complex values')
        with assert_raises(ValueError, match=msg):
            convolve2d([[1]], [[1, 2]], fillvalue=1j)
    msg = '`fillvalue` must be scalar or an array with '
    with assert_raises(ValueError, match=msg):
        convolve2d([[1]], [[1, 2]], fillvalue=[1, 2])