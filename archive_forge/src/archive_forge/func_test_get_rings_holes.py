import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_rings_holes():
    rings = shapely.get_rings(polygon_with_hole)
    assert len(rings) == 2
    assert rings[0] == shapely.get_exterior_ring(polygon_with_hole)
    assert rings[1] == shapely.get_interior_ring(polygon_with_hole, 0)