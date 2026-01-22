import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_num_interior_rings():
    actual = shapely.get_num_interior_rings(all_types + (polygon_with_hole, None))
    assert actual.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]