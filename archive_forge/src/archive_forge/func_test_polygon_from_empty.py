import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_polygon_from_empty():
    polygon = Polygon()
    assert polygon.is_empty
    assert polygon.exterior.coords[:] == []
    polygon = Polygon([])
    assert polygon.is_empty
    assert polygon.exterior.coords[:] == []