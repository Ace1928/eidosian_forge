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
def test_input_swapping(self):
    small = arange(8).reshape(2, 2, 2)
    big = 1j * arange(27).reshape(3, 3, 3)
    big += arange(27)[::-1].reshape(3, 3, 3)
    out_array = array([[[0 + 0j, 26 + 0j, 25 + 1j, 24 + 2j], [52 + 0j, 151 + 5j, 145 + 11j, 93 + 11j], [46 + 6j, 133 + 23j, 127 + 29j, 81 + 23j], [40 + 12j, 98 + 32j, 93 + 37j, 54 + 24j]], [[104 + 0j, 247 + 13j, 237 + 23j, 135 + 21j], [282 + 30j, 632 + 96j, 604 + 124j, 330 + 86j], [246 + 66j, 548 + 180j, 520 + 208j, 282 + 134j], [142 + 66j, 307 + 161j, 289 + 179j, 153 + 107j]], [[68 + 36j, 157 + 103j, 147 + 113j, 81 + 75j], [174 + 138j, 380 + 348j, 352 + 376j, 186 + 230j], [138 + 174j, 296 + 432j, 268 + 460j, 138 + 278j], [70 + 138j, 145 + 323j, 127 + 341j, 63 + 197j]], [[32 + 72j, 68 + 166j, 59 + 175j, 30 + 100j], [68 + 192j, 139 + 433j, 117 + 455j, 57 + 255j], [38 + 222j, 73 + 499j, 51 + 521j, 21 + 291j], [12 + 144j, 20 + 318j, 7 + 331j, 0 + 182j]]])
    assert_array_equal(convolve(small, big, 'full'), out_array)
    assert_array_equal(convolve(big, small, 'full'), out_array)
    assert_array_equal(convolve(small, big, 'same'), out_array[1:3, 1:3, 1:3])
    assert_array_equal(convolve(big, small, 'same'), out_array[0:3, 0:3, 0:3])
    assert_array_equal(convolve(small, big, 'valid'), out_array[1:3, 1:3, 1:3])
    assert_array_equal(convolve(big, small, 'valid'), out_array[1:3, 1:3, 1:3])