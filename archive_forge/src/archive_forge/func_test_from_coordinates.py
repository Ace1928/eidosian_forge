import numpy as np
import pytest
from shapely import Point
from shapely.coords import CoordinateSequence
from shapely.errors import DimensionError
def test_from_coordinates():
    p = Point(1.0, 2.0)
    assert p.coords[:] == [(1.0, 2.0)]
    assert p.has_z is False
    p = Point(1.0, 2.0, 3.0)
    assert p.coords[:] == [(1.0, 2.0, 3.0)]
    assert p.has_z
    p = Point()
    assert p.is_empty
    assert isinstance(p.coords, CoordinateSequence)
    assert p.coords[:] == []