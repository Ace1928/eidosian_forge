from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.parametrize('func', [fft, ifft, fftn, ifftn, rfft, irfft, rfftn, irfftn])
def test_invalid_norm(func):
    x = np.arange(10, dtype=float)
    with assert_raises(ValueError, match='Invalid norm value \'o\', should be "backward", "ortho" or "forward"'):
        func(x, norm='o')