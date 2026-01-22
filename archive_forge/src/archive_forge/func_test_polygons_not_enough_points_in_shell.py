import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('shape', [(2, 1, 2), (1, 1, 2), (1, 2), (2, 2, 2), (1, 2, 2), (2, 2)])
def test_polygons_not_enough_points_in_shell(shape):
    coords = np.ones(shape)
    with pytest.raises(ValueError):
        shapely.polygons(coords)
    coords[..., 1] += 1
    with pytest.raises(ValueError):
        shapely.polygons(coords)