import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_ellipse_perimeter_flat_zeroangle():
    img = np.zeros((20, 18), 'uint8')
    img_ = np.zeros((20, 18), 'uint8')
    rr, cc = ellipse_perimeter(6, 7, 0, 5, 0)
    img[rr, cc] = 1
    rr, cc = line(6, 2, 6, 12)
    img_[rr, cc] = 1
    assert_array_equal(img, img_)