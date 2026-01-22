import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_empty_equality(self):
    point1 = Point(0, 0)
    polygon1 = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
    polygon2 = Polygon([(0.0, 0.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)])
    polygon_empty1 = Polygon()
    polygon_empty2 = Polygon()
    assert point1 != polygon1
    assert polygon_empty1 == polygon_empty2
    assert polygon1 != polygon_empty1
    assert polygon1 == polygon2
    assert polygon_empty1 is not None