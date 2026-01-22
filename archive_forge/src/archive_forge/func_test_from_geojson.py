import numpy as np
import pytest
from shapely import GeometryCollection, LineString, Point, wkt
from shapely.geometry import shape
def test_from_geojson(geometrycollection_geojson):
    geom = shape(geometrycollection_geojson)
    assert geom.geom_type == 'GeometryCollection'
    assert len(geom.geoms) == 2
    geom_types = [g.geom_type for g in geom.geoms]
    assert 'Point' in geom_types
    assert 'LineString' in geom_types