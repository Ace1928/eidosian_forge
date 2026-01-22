import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
def test_ellipse_trivial():
    img = np.zeros((2, 2), 'uint8')
    rr, cc = ellipse(0.5, 0.5, 0.5, 0.5)
    img[rr, cc] = 1
    img_correct = np.array([[0, 0], [0, 0]])
    assert_array_equal(img, img_correct)
    img = np.zeros((2, 2), 'uint8')
    rr, cc = ellipse(0.5, 0.5, 1.1, 1.1)
    img[rr, cc] = 1
    img_correct = np.array([[1, 1], [1, 1]])
    assert_array_equal(img, img_correct)
    img = np.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 0.9, 0.9)
    img[rr, cc] = 1
    img_correct = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert_array_equal(img, img_correct)
    img = np.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 1.1, 1.1)
    img[rr, cc] = 1
    img_correct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    assert_array_equal(img, img_correct)
    img = np.zeros((3, 3), 'uint8')
    rr, cc = ellipse(1, 1, 1.5, 1.5)
    img[rr, cc] = 1
    img_correct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    assert_array_equal(img, img_correct)