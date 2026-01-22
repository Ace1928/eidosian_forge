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
def test_set_precision_drop_coords():
    geometry = shapely.set_precision(LineString([(0, 0), (0, 0), (0, 1), (1, 1)]), 0)
    assert_geometries_equal(geometry, LineString([(0, 0), (0, 0), (0, 1), (1, 1)]))
    geometry = shapely.set_precision(geometry, 1)
    assert_geometries_equal(geometry, LineString([(0, 0), (0, 1), (1, 1)]))