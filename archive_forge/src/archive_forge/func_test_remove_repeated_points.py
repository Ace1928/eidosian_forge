import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason='GEOS < 3.11')
@pytest.mark.parametrize('geom,expected', [(LineString([(0, 0), (0, 0), (1, 0)]), LineString([(0, 0), (1, 0)])), (LinearRing([(0, 0), (1, 2), (1, 2), (1, 3), (0, 0)]), LinearRing([(0, 0), (1, 2), (1, 3), (0, 0)])), (Polygon([(0, 0), (0, 0), (1, 0), (1, 1), (1, 0), (0, 0)]), Polygon([(0, 0), (1, 0), (1, 1), (1, 0), (0, 0)])), (Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)], holes=[[(2, 2), (2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]]), Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)], holes=[[(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)]])), (MultiPolygon([Polygon([(0, 0), (0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), Polygon([(2, 2), (2, 2), (2, 3), (3, 3), (3, 2), (2, 2)])]), MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), Polygon([(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)])])), (point, point), (point_z, point_z), (multi_point, multi_point), (empty_point, empty_point), (empty_line_string, empty_line_string), (empty, empty), (empty_polygon, empty_polygon)])
def test_remove_repeated_points(geom, expected):
    assert_geometries_equal(shapely.remove_repeated_points(geom, 0), expected)