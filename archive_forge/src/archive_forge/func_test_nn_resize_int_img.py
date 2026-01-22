import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
from scipy.ndimage import map_coordinates
from skimage._shared.testing import expected_warnings, run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color.colorconv import rgb2gray
from skimage.data import checkerboard, astronaut
from skimage.draw.draw import circle_perimeter_aa
from skimage.feature.peak import peak_local_max
from skimage.transform._warps import (
from skimage.transform._geometric import (
from skimage.util.dtype import img_as_float, _convert
def test_nn_resize_int_img():
    """Issue #6467"""
    img = np.zeros((12, 12), dtype=np.int16)
    img[4:8, 1:4] = 5
    img[4:8, 7:10] = 7
    resized = resize(img, (8, 8), order=0)
    assert np.array_equal(np.unique(resized), np.unique(img))