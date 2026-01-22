import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_hough_line_peaks_single_line():
    img = np.zeros((100, 100), dtype=bool)
    img[30, :] = 1
    hough_space, angles, dist = transform.hough_line(img)
    best_h_space, best_angles, best_dist = transform.hough_line_peaks(hough_space, angles, dist)
    assert len(best_angles) == 1
    assert len(best_dist) == 1
    expected_angle = -np.pi / 2
    expected_dist = -30
    assert abs(best_angles[0] - expected_angle) < 0.01
    assert abs(best_dist[0] - expected_dist) < 0.01