import numpy as np
import pytest
from shapely import LinearRing, LineString, Point, Polygon
from shapely.coords import CoordinateSequence
from shapely.errors import TopologicalError
from shapely.wkb import loads as load_wkb
def test_polygon_from_coordinate_sequence_with_holes():
    coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    polygon = Polygon(coords, [[(0.25, 0.25), (0.25, 0.5), (0.5, 0.5), (0.5, 0.25)]])
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 1
    assert len(polygon.interiors[0].coords) == 5
    coords = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
    holes = [[(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)], [(3, 3), (3, 4), (4, 5), (5, 4), (5, 3), (3, 3)]]
    polygon = Polygon(coords, holes)
    assert polygon.exterior.coords[:] == coords
    assert len(polygon.interiors) == 2
    assert len(polygon.interiors[0].coords) == 5
    assert len(polygon.interiors[1].coords) == 6