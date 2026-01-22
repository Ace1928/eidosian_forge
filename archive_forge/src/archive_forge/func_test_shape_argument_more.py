from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_shape_argument_more(self):
    x = zeros((4, 4, 2))
    with assert_raises(ValueError, match='shape requires more axes than are present'):
        fftn(x, s=(8, 8, 2, 1))