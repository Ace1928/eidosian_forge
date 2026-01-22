import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linearrings_invalid_ndim():
    msg = 'The ordinate \\(last\\) dimension should be 2 or 3, got {}'
    coords1 = np.random.randn(10, 3, 4)
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linearrings(coords1)
    coords2 = np.hstack((coords1, coords1[:, [0], :]))
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linearrings(coords2)
    coords3 = np.random.randn(10, 3, 1)
    with pytest.raises(ValueError, match=msg.format(1)):
        shapely.linestrings(coords3)