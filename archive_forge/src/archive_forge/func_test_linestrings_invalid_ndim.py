import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linestrings_invalid_ndim():
    msg = 'The ordinate \\(last\\) dimension should be 2 or 3, got {}'
    coords = np.ones((10, 2, 4), order='C')
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linestrings(coords)
    coords = np.ones((10, 2, 4), order='F')
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linestrings(coords)
    coords = np.swapaxes(np.swapaxes(np.ones((10, 2, 4)), 0, 2), 1, 0)
    coords = np.swapaxes(np.swapaxes(np.asarray(coords, order='F'), 0, 2), 1, 2)
    with pytest.raises(ValueError, match=msg.format(4)):
        shapely.linestrings(coords)
    coords = np.ones((10, 2, 1))
    with pytest.raises(ValueError, match=msg.format(1)):
        shapely.linestrings(coords)