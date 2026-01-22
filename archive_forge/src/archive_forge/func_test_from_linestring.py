import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_from_linestring():
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    copy = LineString(line)
    assert copy.coords[:] == [(1.0, 2.0), (3.0, 4.0)]
    assert copy.geom_type == 'LineString'