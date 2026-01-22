import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type
def test_hog_visualization_orientation():
    """Test that the visualization produces a line with correct orientation

    The hog visualization is expected to draw line segments perpendicular to
    the midpoints of orientation bins.  This example verifies that when
    orientations=3 and the gradient is entirely in the middle bin (bisected
    by the y-axis), the line segment drawn by the visualization is horizontal.
    """
    width = height = 11
    image = np.zeros((height, width), dtype='float')
    image[height // 2:] = 1
    _, hog_image = feature.hog(image, orientations=3, pixels_per_cell=(width, height), cells_per_block=(1, 1), visualize=True, block_norm='L1')
    middle_index = height // 2
    indices_excluding_middle = [x for x in range(height) if x != middle_index]
    assert (hog_image[indices_excluding_middle, :] == 0).all()
    assert (hog_image[middle_index, 1:-1] > 0).all()