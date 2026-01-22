from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def test_cc_diff(self):
    for dtype in self.dtypes:
        self._check_1d(cc_diff, dtype, (16,), 1.0, 4.0)