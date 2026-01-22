from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def test_zero_nyquist(self):
    for k in [0, 1, 2, 3, 4, 5, 6]:
        for n in [32, 33, 64, 56, 55]:
            f = random((n,))
            af = sum(f, axis=0) / n
            f = f - af
            f = diff(diff(f, 1), -1)
            assert_almost_equal(sum(f, axis=0), 0.0)
            assert_array_almost_equal(diff(diff(f, k), -k), f)
            assert_array_almost_equal(diff(diff(f, -k), k), f)