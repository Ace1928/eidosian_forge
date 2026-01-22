import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def check_hough_line_peaks_angle():
    img = np.zeros((100, 100), dtype=bool)
    img[:, 0] = True
    img[0, :] = True
    hspace, angles, dists = transform.hough_line(img)
    assert len(transform.hough_line_peaks(hspace, angles, dists, min_angle=45)[0]) == 2
    assert len(transform.hough_line_peaks(hspace, angles, dists, min_angle=90)[0]) == 1
    theta = np.linspace(0, np.pi, 100)
    hspace, angles, dists = transform.hough_line(img, theta)
    assert len(transform.hough_line_peaks(hspace, angles, dists, min_angle=45)[0]) == 2
    assert len(transform.hough_line_peaks(hspace, angles, dists, min_angle=90)[0]) == 1
    theta = np.linspace(np.pi / 3, 4.0 / 3 * np.pi, 100)
    hspace, angles, dists = transform.hough_line(img, theta)
    assert len(transform.hough_line_peaks(hspace, angles, dists, min_angle=45)[0]) == 2
    assert len(transform.hough_line_peaks(hspace, angles, dists, min_angle=90)[0]) == 1