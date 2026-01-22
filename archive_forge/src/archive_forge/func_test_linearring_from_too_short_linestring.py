import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_linearring_from_too_short_linestring():
    coords = [(0.0, 0.0), (1.0, 1.0)]
    line = LineString(coords)
    with pytest.raises(ValueError, match='requires at least 4 coordinates'):
        LinearRing(line)