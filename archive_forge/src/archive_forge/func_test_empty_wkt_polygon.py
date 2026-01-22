import unittest
from shapely import wkt
from shapely.geometry import shape
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import LinearRing, Polygon
def test_empty_wkt_polygon():
    """Confirm fix for issue #450"""
    g = wkt.loads('POLYGON EMPTY')
    assert g.__geo_interface__['type'] == 'Polygon'
    assert g.__geo_interface__['coordinates'] == ()