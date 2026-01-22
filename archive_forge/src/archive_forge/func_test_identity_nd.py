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
@pytest.mark.parametrize('forward, backward', [(dctn, idctn), (dstn, idstn)])
@pytest.mark.parametrize('type', [1, 2, 3, 4])
@pytest.mark.parametrize('shape, axes', [((4, 4), 0), ((4, 4), 1), ((4, 4), None), ((4, 4), (0, 1)), ((10, 12), None), ((10, 12), (0, 1)), ((4, 5, 6), None), ((4, 5, 6), 1), ((4, 5, 6), (0, 2))])
@pytest.mark.parametrize('norm', [None, 'backward', 'ortho', 'forward'])
@pytest.mark.parametrize('orthogonalize', [False, True])
def test_identity_nd(forward, backward, type, shape, axes, norm, orthogonalize, xp):
    x = xp.asarray(np.random.random(shape))
    if axes is not None:
        shape = np.take(shape, axes)
    y = forward(x, type, axes=axes, norm=norm, orthogonalize=orthogonalize)
    z = backward(y, type, axes=axes, norm=norm, orthogonalize=orthogonalize)
    xp_assert_close(z, x)
    if axes is None:
        pad = [(0, 4)] * x.ndim
    elif isinstance(axes, int):
        pad = [(0, 0)] * x.ndim
        pad[axes] = (0, 4)
    else:
        pad = [(0, 0)] * x.ndim
        for a in axes:
            pad[a] = (0, 4)
    y2 = xp.asarray(np.pad(np.asarray(y), pad, mode='edge'))
    z2 = backward(y2, type, shape, axes, norm, orthogonalize=orthogonalize)
    xp_assert_close(z2, x)