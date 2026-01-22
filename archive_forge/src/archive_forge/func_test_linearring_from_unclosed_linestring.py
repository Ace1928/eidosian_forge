import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring_from_unclosed_linestring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    line = LineString(coords[:-1])
    ring = LinearRing(line)
    assert len(ring.coords) == 4
    assert ring.coords[:] == coords
    assert ring.geom_type == 'LinearRing'