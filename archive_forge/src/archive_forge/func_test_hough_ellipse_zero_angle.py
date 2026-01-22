import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_ellipse_zero_angle():
    img = np.zeros((25, 25), dtype=int)
    rx = 6
    ry = 8
    x0 = 12
    y0 = 15
    angle = 0
    rr, cc = ellipse_perimeter(y0, x0, ry, rx)
    img[rr, cc] = 1
    result = transform.hough_ellipse(img, threshold=9)
    best = result[-1]
    assert_equal(best[1], y0)
    assert_equal(best[2], x0)
    assert_almost_equal(best[3], ry, decimal=1)
    assert_almost_equal(best[4], rx, decimal=1)
    assert_equal(best[5], angle)
    rr2, cc2 = ellipse_perimeter(y0, x0, int(best[3]), int(best[4]), orientation=best[5])
    assert_equal(rr, rr2)
    assert_equal(cc, cc2)