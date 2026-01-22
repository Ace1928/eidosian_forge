import unittest
import pytest
import shapely
from shapely.errors import TopologicalError
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.wkt import loads
def test_relate(self):
    assert Point(0, 0).relate(Point(-1, -1)) == 'FF0FFF0F2'
    invalid_polygon = loads('POLYGON ((40 100, 80 100, 80 60, 40 60, 40 100), (60 60, 80 60, 80 40, 60 40, 60 60))')
    assert not invalid_polygon.is_valid
    with pytest.raises((TopologicalError, shapely.GEOSException)):
        invalid_polygon.relate(invalid_polygon)