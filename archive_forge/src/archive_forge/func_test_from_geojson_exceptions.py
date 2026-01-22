import json
import pickle
import struct
import warnings
import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LineString, Point, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types, empty_point, empty_point_z, point, point_z
@pytest.mark.skipif(shapely.geos_version < (3, 10, 1), reason='GEOS < 3.10.1')
def test_from_geojson_exceptions():
    with pytest.raises(TypeError, match='Expected bytes or string, got int'):
        shapely.from_geojson(1)
    with pytest.raises(shapely.GEOSException, match='Error parsing JSON'):
        shapely.from_geojson('')
    with pytest.raises(shapely.GEOSException, match='Unknown geometry type'):
        shapely.from_geojson('{"type": "NoGeometry", "coordinates": []}')
    with pytest.raises(shapely.GEOSException, match='type must be array, but is null'):
        shapely.from_geojson('{"type": "LineString", "coordinates": null}')
    with pytest.raises(shapely.GEOSException, match="key 'type' not found"):
        shapely.from_geojson('{"geometry": null, "properties": []}')
    with pytest.raises(shapely.GEOSException, match="key 'type' not found"):
        shapely.from_geojson('{"no": "geojson"}')