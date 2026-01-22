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
def cast_tf2sos(b, a):
    """Convert TF2SOS, casting to complex128 and back to the original dtype."""
    dtype = np.asarray(b).dtype
    b = np.array(b, np.complex128)
    a = np.array(a, np.complex128)
    return tf2sos(b, a).astype(dtype)