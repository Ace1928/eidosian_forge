import numpy as np
import pytest
from numpy.testing import assert_equal
from skimage._shared.testing import fetch
from skimage.morphology import footprints
def test_cube_footprint(self):
    """Test cube footprints"""
    for k in range(0, 5):
        actual_mask = footprints.cube(k)
        expected_mask = np.ones((k, k, k), dtype='uint8')
        assert_equal(expected_mask, actual_mask)