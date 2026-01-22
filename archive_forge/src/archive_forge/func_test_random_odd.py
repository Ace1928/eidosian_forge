from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def test_random_odd(self):
    for n in [33, 65, 55]:
        f = random((n,))
        af = sum(f, axis=0) / n
        f = f - af
        assert_almost_equal(sum(f, axis=0), 0.0)
        assert_array_almost_equal(ihilbert(hilbert(f)), f)
        assert_array_almost_equal(hilbert(ihilbert(f)), f)