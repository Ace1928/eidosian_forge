import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
def test_2d_ndimage_equivalence():
    image = np.zeros((9, 9), np.uint8)
    image[2:-2, 2:-2] = 128
    image[3:-3, 3:-3] = 196
    image[4, 4] = 255
    opened = gray.opening(image)
    closed = gray.closing(image)
    footprint = ndi.generate_binary_structure(2, 1)
    ndimage_opened = ndi.grey_opening(image, footprint=footprint)
    ndimage_closed = ndi.grey_closing(image, footprint=footprint)
    assert_array_equal(opened, ndimage_opened)
    assert_array_equal(closed, ndimage_closed)