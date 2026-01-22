import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
def test_hog_odd_cell_size():
    img = np.zeros((3, 3))
    img[2, 2] = 1
    correct_output = np.zeros((9,))
    correct_output[0] = 0.5
    correct_output[4] = 0.5
    output = feature.hog(img, pixels_per_cell=(3, 3), cells_per_block=(1, 1), block_norm='L1')
    assert_almost_equal(output, correct_output, decimal=1)