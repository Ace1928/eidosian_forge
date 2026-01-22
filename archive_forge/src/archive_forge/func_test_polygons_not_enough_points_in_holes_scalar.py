import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygons_not_enough_points_in_holes_scalar():
    with pytest.raises(ValueError):
        shapely.polygons(np.ones((1, 4, 2)), (1, 1))