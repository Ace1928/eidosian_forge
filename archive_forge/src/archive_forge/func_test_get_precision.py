import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 6, 0), reason='GEOS < 3.6')
def test_get_precision():
    geometries = all_types + (point_z, empty_point, empty_line_string, empty_polygon)
    actual = shapely.get_precision(geometries).tolist()
    assert actual == [0] * len(geometries)
    geometry = shapely.set_precision(geometries, 1)
    actual = shapely.get_precision(geometry).tolist()
    assert actual == [1] * len(geometries)