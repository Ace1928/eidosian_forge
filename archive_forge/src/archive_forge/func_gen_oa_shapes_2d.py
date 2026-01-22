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
def gen_oa_shapes_2d(sizes):
    shapes0 = gen_oa_shapes(sizes)
    shapes1 = gen_oa_shapes(sizes)
    shapes = [ishapes0 + ishapes1 for ishapes0, ishapes1 in zip(shapes0, shapes1)]
    modes = ['full', 'valid', 'same']
    return [ishapes + (imode,) for ishapes, imode in product(shapes, modes) if imode != 'valid' or (ishapes[0] > ishapes[1] and ishapes[2] > ishapes[3]) or (ishapes[0] < ishapes[1] and ishapes[2] < ishapes[3])]