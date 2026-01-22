import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_from_coordinate_sequence():
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    assert len(line.coords) == 2
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]