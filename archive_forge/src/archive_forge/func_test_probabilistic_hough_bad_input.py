import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal
from skimage import data, transform
from skimage._shared.testing import run_in_parallel
from skimage.draw import circle_perimeter, ellipse_perimeter, line
def test_probabilistic_hough_bad_input():
    img = np.zeros(100)
    img[10] = 1
    with pytest.raises(ValueError):
        transform.probabilistic_hough_line(img)