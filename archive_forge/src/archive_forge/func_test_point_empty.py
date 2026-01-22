import numpy as np
import pytest
from shapely import Point
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError
def test_point_empty(self):
    p_null = Point()
    assert p_null.wkt == 'POINT EMPTY'
    assert p_null.coords[:] == []
    assert p_null.area == 0.0