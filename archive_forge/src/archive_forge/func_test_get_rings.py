import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.parametrize('geom', [point, multi_point, line_string, multi_line_string, polygon, multi_polygon, geometry_collection, empty_point, empty_line_string, empty_polygon, empty_geometry_collection, None])
def test_get_rings(geom):
    if shapely.get_type_id(geom) != shapely.GeometryType.POLYGON or shapely.is_empty(geom):
        rings = shapely.get_rings(geom)
        assert len(rings) == 0
    else:
        rings = shapely.get_rings(geom)
        assert len(rings) == 1
        assert rings[0] == shapely.get_exterior_ring(geom)