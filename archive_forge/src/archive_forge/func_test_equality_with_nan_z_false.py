import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
def test_equality_with_nan_z_false():
    with ignore_invalid():
        left = LineString([(0, 1, np.nan), (2, 3, np.nan)])
        right = LineString([(0, 1, np.nan), (2, 3, 4)])
    if shapely.geos_version < (3, 10, 0):
        assert left == right
    elif shapely.geos_version < (3, 12, 0):
        assert left == right
    else:
        assert left != right