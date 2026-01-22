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
def test_spaced_1dperiod(self):
    events = np.array([0.1, 1.1, 2.1, 4.1, 10.1])
    period = 1
    targ_strength = 1.0
    targ_phase = 0.1
    strength, phase = vectorstrength(events, period)
    assert_equal(strength.ndim, 0)
    assert_equal(phase.ndim, 0)
    assert_almost_equal(strength, targ_strength)
    assert_almost_equal(phase, 2 * np.pi * targ_phase)