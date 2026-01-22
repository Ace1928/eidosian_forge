import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_from_empty():
    line = LineString()
    assert line.is_empty
    assert isinstance(line.coords, CoordinateSequence)
    assert line.coords[:] == []
    line = LineString([])
    assert line.is_empty
    assert isinstance(line.coords, CoordinateSequence)
    assert line.coords[:] == []