import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
from skimage._shared.testing import run_in_parallel
from skimage._shared._dependency_checks import has_mpl
from skimage.draw import (
from skimage.measure import regionprops
@pytest.mark.skipif(not has_mpl, reason='matplotlib not installed')
def test_polygon_perimeter_outside_image():
    rr, cc = polygon_perimeter([-1, -1, 3, 3], [-1, 4, 4, -1], shape=(3, 4))
    assert_equal(len(rr), 0)
    assert_equal(len(cc), 0)