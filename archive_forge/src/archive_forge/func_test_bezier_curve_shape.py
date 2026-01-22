import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_bezier_curve_shape():
    img = np.zeros((15, 20), 'uint8')
    r0, c0 = (1, 5)
    r1, c1 = (6, 11)
    r2, c2 = (1, 14)
    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 2, shape=(15, 20))
    img[rr, cc] = 1
    shift = 5
    img_ = np.zeros((15 + 2 * shift, 20), 'uint8')
    r0, c0 = (1 + shift, 5)
    r1, c1 = (6 + shift, 11)
    r2, c2 = (1 + shift, 14)
    rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 2, shape=None)
    img_[rr, cc] = 1
    assert_array_equal(img, img_[shift:-shift, :])