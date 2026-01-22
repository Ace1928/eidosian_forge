import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_line_angles():
    img = np.zeros((10, 10))
    img[0, 0] = 1
    out, angles, d = transform.hough_line(img, np.linspace(0, 360, 10))
    assert_equal(len(angles), 10)