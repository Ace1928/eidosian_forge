from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
@pytest.mark.parametrize('dtype', real_dtypes)
@pytest.mark.parametrize('fftsize', fftsizes)
@pytest.mark.parametrize('overwrite_x', [True, False])
@pytest.mark.parametrize('shape,axes', [((16,), -1), ((16, 2), 0), ((2, 16), 1)])
def test_rfft_irfft(self, dtype, fftsize, overwrite_x, shape, axes):
    overwritable = self.real_dtypes
    self._check_1d(irfft, dtype, shape, axes, overwritable, fftsize, overwrite_x)
    self._check_1d(rfft, dtype, shape, axes, overwritable, fftsize, overwrite_x)