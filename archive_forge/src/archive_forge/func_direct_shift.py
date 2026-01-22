from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.fftpack import (diff, fft, ifft, tilbert, itilbert, hilbert,
import numpy as np
from numpy import arange, sin, cos, pi, exp, tanh, sum, sign
from numpy.random import random
def direct_shift(x, a, period=None):
    n = len(x)
    if period is None:
        k = fftfreq(n) * 1j * n
    else:
        k = fftfreq(n) * 2j * pi / period * n
    return ifft(fft(x) * exp(k * a)).real