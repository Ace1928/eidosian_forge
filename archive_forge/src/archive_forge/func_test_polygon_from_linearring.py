import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_polygon_from_linearring():
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    ring = LinearRing(coords)
    polygon = Polygon(ring)
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 0
    shell = LinearRing([(0.0, 0.0), (70.0, 120.0), (140.0, 0.0), (0.0, 0.0)])
    holes = [LinearRing([(60.0, 80.0), (80.0, 80.0), (70.0, 60.0), (60.0, 80.0)]), LinearRing([(30.0, 10.0), (50.0, 10.0), (40.0, 30.0), (30.0, 10.0)]), LinearRing([(90.0, 10), (110.0, 10.0), (100.0, 30.0), (90.0, 10.0)])]
    polygon = Polygon(shell, holes)
    assert polygon.exterior.coords[:] == shell.coords[:]
    assert len(polygon.interiors) == 3
    for i in range(3):
        assert polygon.interiors[i].coords[:] == holes[i].coords[:]