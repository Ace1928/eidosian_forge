import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygon_wrong_hole_type():
    with pytest.raises((TypeError, shapely.GEOSException)):
        shapely.polygons(linear_ring, [point])