import numpy as np
import pytest
from scipy import ndimage as ndi
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from skimage import color, data, transform
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch, assert_stacklevel
from skimage.morphology import gray, footprints
from skimage.util import img_as_uint, img_as_ubyte
def test_gray_opening_anti_extensive(self):
    img = data.coins()
    footprint = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    result_default = gray.opening(img, footprint=footprint)
    assert not np.all(result_default <= img)
    result_ignore = gray.opening(img, footprint=footprint, mode='ignore')
    assert np.all(result_ignore <= img)