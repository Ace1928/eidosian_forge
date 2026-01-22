import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiLineString, Point, Polygon
from shapely.tests.common import all_types, all_types_z, ignore_invalid
@pytest.mark.parametrize('left, right', cases3)
def test_equality_with_nan_false(left, right):
    assert left != right