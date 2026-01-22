import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_numpy_empty_linearring_coords():
    ring = LinearRing()
    assert np.asarray(ring.coords).shape == (0, 2)