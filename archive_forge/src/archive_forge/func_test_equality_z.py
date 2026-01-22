import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
def test_equality_z():
    geom1 = Point(0, 1)
    geom2 = Point(0, 1, 0)
    assert geom1 != geom2
    geom2 = Point(0, 1, np.nan)
    if shapely.geos_version < (3, 10, 0):
        assert geom1 == geom2
    elif shapely.geos_version < (3, 12, 0):
        assert geom1 == geom2
    else:
        assert geom1 != geom2