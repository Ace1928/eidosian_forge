import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_linestring_immutable():
    line = LineString([(1.0, 2.0), (3.0, 4.0)])
    with pytest.raises(AttributeError):
        line.coords = [(-1.0, -1.0), (1.0, 1.0)]
    with pytest.raises(TypeError):
        line.coords[0] = (-1.0, -1.0)