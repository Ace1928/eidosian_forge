import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_from_mix():
    line = LineString([Point(1.0, 2.0), (2.0, 3.0), Point(3.0, 4.0)])
    assert line.coords[:] == [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]