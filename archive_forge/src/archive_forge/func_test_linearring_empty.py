import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring_empty(self):
    r_null = LinearRing()
    assert r_null.wkt == 'LINEARRING EMPTY'
    assert r_null.length == 0.0