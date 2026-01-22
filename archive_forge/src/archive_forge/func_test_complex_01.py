import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
def test_complex_01(self):
    data = (1.0 + arange(3.0)) * (1.0 + 1j)
    x = copy(data)
    t = toeplitz(x)
    assert_array_equal(x, data)
    col0 = t[:, 0]
    assert_array_equal(col0, data)
    assert_array_equal(t[0, 1:], data[1:].conj())