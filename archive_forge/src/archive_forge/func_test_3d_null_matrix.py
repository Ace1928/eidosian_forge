import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less, assert_equal
from skimage import img_as_float
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.data import camera, retina
from skimage.filters import frangi, hessian, meijering, sato
from skimage.util import crop, invert
def test_3d_null_matrix():
    a_black = np.zeros((3, 3, 5)).astype(np.uint8)
    a_white = invert(a_black)
    zeros = np.zeros((3, 3, 5))
    ones = np.ones((3, 3, 5))
    assert_allclose(meijering(a_black, black_ridges=True), zeros, atol=0.1)
    assert_allclose(meijering(a_white, black_ridges=False), zeros, atol=0.1)
    assert_equal(sato(a_black, black_ridges=True, mode='reflect'), zeros)
    assert_equal(sato(a_white, black_ridges=False, mode='reflect'), zeros)
    assert_allclose(frangi(a_black, black_ridges=True), zeros, atol=0.001)
    assert_allclose(frangi(a_white, black_ridges=False), zeros, atol=0.001)
    assert_equal(hessian(a_black, black_ridges=False, mode='reflect'), ones)
    assert_equal(hessian(a_white, black_ridges=True, mode='reflect'), ones)