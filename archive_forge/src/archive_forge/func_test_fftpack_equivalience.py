import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.fft import dct, idct, dctn, idctn, dst, idst, dstn, idstn
import scipy.fft as fft
from scipy import fftpack
from scipy.conftest import (
from scipy._lib._array_api import copy, xp_assert_close
import math
@skip_if_array_api_gpu
@array_api_compatible
@pytest.mark.parametrize('func', ['dct', 'dst', 'dctn', 'dstn'])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
@pytest.mark.parametrize('norm', [None, 'backward', 'ortho', 'forward'])
def test_fftpack_equivalience(func, type, norm, xp):
    x = np.random.rand(8, 16)
    fftpack_res = xp.asarray(getattr(fftpack, func)(x, type, norm=norm))
    x = xp.asarray(x)
    fft_res = getattr(fft, func)(x, type, norm=norm)
    xp_assert_close(fft_res, fftpack_res)