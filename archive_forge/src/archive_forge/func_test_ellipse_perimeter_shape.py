import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_ellipse_perimeter_shape():
    img = np.zeros((15, 20), 'uint8')
    rr, cc = ellipse_perimeter(7, 10, 9, 9, 0, shape=(15, 20))
    img[rr, cc] = 1
    shift = 5
    img_ = np.zeros((15 + 2 * shift, 20), 'uint8')
    rr, cc = ellipse_perimeter(7 + shift, 10, 9, 9, 0, shape=None)
    img_[rr, cc] = 1
    assert_array_equal(img, img_[shift:-shift, :])