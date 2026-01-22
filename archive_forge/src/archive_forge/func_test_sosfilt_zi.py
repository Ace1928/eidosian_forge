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
def test_sosfilt_zi(self, dt):
    sos = signal.butter(6, 0.2, output='sos')
    zi = sosfilt_zi(sos)
    y, zf = sosfilt(sos, np.ones(40, dt), zi=zi)
    assert_allclose_cast(zf, zi, rtol=1e-13)
    ss = np.prod(sos[:, :3].sum(axis=-1) / sos[:, 3:].sum(axis=-1))
    assert_allclose_cast(y, ss, rtol=1e-13)
    _, zf = sosfilt(sos, np.ones(40, dt), zi=zi.tolist())
    assert_allclose_cast(zf, zi, rtol=1e-13)