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
def test_complex_fir_dlti(self):
    fcentre = 50
    fwidth = 5
    fs = 1000.0
    numtaps = 20
    bbase = signal.firwin(numtaps, fwidth / 2, fs=fs)
    zbase = np.roots(bbase)
    zrot = zbase * np.exp(2j * np.pi * fcentre / fs)
    bz = bbase[0] * np.poly(zrot)
    system = signal.dlti(bz, 1)
    t = np.arange(200) / fs
    u = np.exp(2j * np.pi * fcentre * t) + 0.5 * np.exp(-2j * np.pi * fcentre * t)
    ynzp = signal.decimate(u, 2, ftype=system, zero_phase=False)
    ynzpref = signal.upfirdn(bz, u, up=1, down=2)[:100]
    assert_equal(ynzp, ynzpref)
    yzp = signal.decimate(u, 2, ftype=system, zero_phase=True)
    yzpref = signal.resample_poly(u, 1, 2, window=bz)
    assert_equal(yzp, yzpref)