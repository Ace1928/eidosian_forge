from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def direct_rdft(x):
    x = asarray(x)
    n = len(x)
    w = -arange(n) * (2j * pi / n)
    y = zeros(n // 2 + 1, dtype=cdouble)
    for i in range(n // 2 + 1):
        y[i] = dot(exp(i * w), x)
    return y