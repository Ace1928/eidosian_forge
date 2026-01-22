import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from skimage import data, draw, img_as_float
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import run_in_parallel
from skimage._shared.utils import _supported_float_type
from skimage.color import rgb2gray
from skimage.feature import (
from skimage.morphology import cube, octagon
def test_corner_orientations_even_shape_error():
    img = np.zeros((20, 20))
    with pytest.raises(ValueError):
        corner_orientations(img, np.asarray([[7, 7]]), np.ones((4, 4)))