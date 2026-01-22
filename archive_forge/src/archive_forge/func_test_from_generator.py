import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_from_generator():
    gen = (coord for coord in [(1.0, 2.0), (3.0, 4.0)])
    line = LineString(gen)
    assert line.coords[:] == [(1.0, 2.0), (3.0, 4.0)]