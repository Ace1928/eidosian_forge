import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_line_peaks_single_angle():
    img = np.random.random((100, 100))
    tested_angles = np.array([np.pi / 2])
    h, theta, d = transform.hough_line(img, theta=tested_angles)
    accum, angles, dists = transform.hough_line_peaks(h, theta, d, threshold=2)