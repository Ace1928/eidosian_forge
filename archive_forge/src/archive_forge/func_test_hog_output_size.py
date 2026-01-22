import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
def test_hog_output_size():
    img = img_as_float(data.astronaut()[:256, :].mean(axis=2))
    fd = feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L1')
    assert len(fd) == 9 * (256 // 8) * (512 // 8)