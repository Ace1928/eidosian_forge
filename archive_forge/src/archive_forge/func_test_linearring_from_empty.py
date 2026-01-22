import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring_from_empty():
    ring = LinearRing()
    assert ring.is_empty
    assert isinstance(ring.coords, CoordinateSequence)
    assert ring.coords[:] == []
    ring = LinearRing([])
    assert ring.is_empty
    assert isinstance(ring.coords, CoordinateSequence)
    assert ring.coords[:] == []