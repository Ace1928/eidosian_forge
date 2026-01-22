from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_non_ndarray_with_dtype(self):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    xs = _TestRFFTBase.MockSeries(x)
    expected = [1, 2, 3, 4, 5]
    rfft(xs)
    assert_equal(x, expected)
    assert_equal(xs.data, expected)