from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def test_expr_large(self):
    for n in [2048, 4096]:
        x = arange(n) * 2 * pi / n
        f = sin(x) * cos(4 * x) + exp(sin(3 * x))
        df = cos(x) * cos(4 * x) - 4 * sin(x) * sin(4 * x) + 3 * cos(3 * x) * exp(sin(3 * x))
        ddf = -17 * sin(x) * cos(4 * x) - 8 * cos(x) * sin(4 * x) - 9 * sin(3 * x) * exp(sin(3 * x)) + 9 * cos(3 * x) ** 2 * exp(sin(3 * x))
        assert_array_almost_equal(diff(f), df)
        assert_array_almost_equal(diff(df), ddf)
        assert_array_almost_equal(diff(ddf, -1), df)
        assert_array_almost_equal(diff(f, 2), ddf)