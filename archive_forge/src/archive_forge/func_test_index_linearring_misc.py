import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_index_linearring_misc(self):
    g = Polygon()
    with pytest.raises(IndexError):
        g.interiors[0]
    with pytest.raises(TypeError):
        g.interiors[0.0]