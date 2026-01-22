import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_rectangle_extent():
    expected = np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)
    start = (1, 1)
    extent = (3, 3)
    img = np.zeros((5, 5), dtype=np.uint8)
    rr, cc = rectangle(start, extent=extent, shape=img.shape)
    img[rr, cc] = 1
    assert_array_equal(img, expected)
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    rr, cc = rectangle(start, extent=extent, shape=img.shape)
    img[rr, cc, 0] = 1
    expected_2 = np.zeros_like(img)
    expected_2[..., 0] = expected
    assert_array_equal(img, expected_2)