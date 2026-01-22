import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_get_dimensions():
    actual = shapely.get_dimensions(all_types).tolist()
    assert actual == [0, 1, 1, 2, 0, 1, 2, 1, -1]