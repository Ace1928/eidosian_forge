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
def test_warp_identity():
    img = img_as_float(rgb2gray(astronaut()))
    assert len(img.shape) == 2
    assert np.allclose(img, warp(img, AffineTransform(rotation=0)))
    assert not np.allclose(img, warp(img, AffineTransform(rotation=0.1)))
    rgb_img = np.transpose(np.asarray([img, np.zeros_like(img), img]), (1, 2, 0))
    warped_rgb_img = warp(rgb_img, AffineTransform(rotation=0.1))
    assert np.allclose(rgb_img, warp(rgb_img, AffineTransform(rotation=0)))
    assert not np.allclose(rgb_img, warped_rgb_img)
    assert np.all(0 == warped_rgb_img[:, :, 1])