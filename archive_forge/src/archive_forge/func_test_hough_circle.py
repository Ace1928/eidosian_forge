import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
@run_in_parallel()
def test_hough_circle():
    img = np.zeros((120, 100), dtype=int)
    radius = 20
    x_0, y_0 = (99, 50)
    y, x = circle_perimeter(y_0, x_0, radius)
    img[x, y] = 1
    out1 = transform.hough_circle(img, radius)
    out2 = transform.hough_circle(img, [radius])
    assert_equal(out1, out2)
    out = transform.hough_circle(img, np.array([radius], dtype=np.intp))
    assert_equal(out, out1)
    x, y = np.where(out[0] == out[0].max())
    assert_equal(x[0], x_0)
    assert_equal(y[0], y_0)