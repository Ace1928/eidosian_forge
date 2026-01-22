from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.parametrize('func', [ihfft, ihfftn, rfft, rfftn])
def test_swapped_byte_order_real(func):
    rng = np.random.RandomState(1234)
    x = rng.rand(10)
    assert_allclose(func(swap_byteorder(x)), func(x))