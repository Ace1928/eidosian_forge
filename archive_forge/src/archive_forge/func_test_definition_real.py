from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_definition_real(self):
    x = np.array([1, 2, 3, 4, 1, 2, 3, 4], self.rdt)
    y = ifft(x)
    assert_equal(y.dtype, self.cdt)
    y1 = direct_idft(x)
    assert_array_almost_equal(y, y1)
    x = np.array([1, 2, 3, 4, 5], dtype=self.rdt)
    assert_equal(y.dtype, self.cdt)
    assert_array_almost_equal(ifft(x), direct_idft(x))