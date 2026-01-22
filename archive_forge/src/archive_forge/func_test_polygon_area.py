import pytest
from skimage._shared._geometry import polygon_clip, polygon_area
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
def test_polygon_area():
    x = [0, 0, 1, 1]
    y = [0, 1, 1, 0]
    assert_almost_equal(polygon_area(y, x), 1)
    x = [0, 0, 1]
    y = [0, 1, 1]
    assert_almost_equal(polygon_area(y, x), 0.5)
    x = [0, 0, 0.5, 1, 1, 0.5]
    y = [0, 1, 0.5, 1, 0, 0.5]
    assert_almost_equal(polygon_area(y, x), 0.5)