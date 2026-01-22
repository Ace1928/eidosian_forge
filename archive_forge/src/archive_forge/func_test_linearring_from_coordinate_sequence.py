import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring_from_coordinate_sequence():
    expected_coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert ring.coords[:] == expected_coords
    ring = LinearRing([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
    assert ring.coords[:] == expected_coords