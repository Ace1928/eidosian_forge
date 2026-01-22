import math
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import scipy.fft as fftmodule
from skimage._shared.utils import _supported_float_type
from skimage.data import astronaut, coins
from skimage.filters import butterworth
from skimage.filters._fft_based import _get_nd_butterworth_filter
def test_butterworth_4D_zeros():
    im = np.zeros((3, 4, 5, 6))
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert_array_equal(im, filtered)