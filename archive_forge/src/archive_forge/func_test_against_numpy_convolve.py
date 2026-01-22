import pytest
import numpy as np
from numpy import arange, add, array, eye, copy, sqrt
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
from scipy.fft import fft
from scipy.special import comb
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,
from numpy.linalg import cond
@pytest.mark.parametrize('cpx', [False, True])
@pytest.mark.parametrize('na', [1, 2, 9])
@pytest.mark.parametrize('nv', [1, 2, 9])
@pytest.mark.parametrize('mode', [None, 'full', 'valid', 'same'])
def test_against_numpy_convolve(self, cpx, na, nv, mode):
    a = self.create_vector(na, cpx)
    v = self.create_vector(nv, cpx)
    if mode is None:
        y1 = np.convolve(v, a)
        A = convolution_matrix(a, nv)
    else:
        y1 = np.convolve(v, a, mode)
        A = convolution_matrix(a, nv, mode)
    y2 = A @ v
    assert_array_almost_equal(y1, y2)