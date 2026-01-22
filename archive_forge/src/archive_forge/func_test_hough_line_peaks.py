import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_line_peaks():
    img = np.zeros((100, 150), dtype=int)
    rr, cc = line(60, 130, 80, 10)
    img[rr, cc] = 1
    out, angles, d = transform.hough_line(img)
    out, theta, dist = transform.hough_line_peaks(out, angles, d)
    assert_equal(len(dist), 1)
    assert_almost_equal(dist[0], 81.0, 1)
    assert_almost_equal(theta[0], 1.41, 1)