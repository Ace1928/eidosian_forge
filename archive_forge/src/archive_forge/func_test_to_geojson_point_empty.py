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
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')
@pytest.mark.parametrize('geom', [empty_point, shapely.multipoints([empty_point, point]), shapely.geometrycollections([empty_point, point]), shapely.geometrycollections([shapely.geometrycollections([empty_point]), point])])
def test_to_geojson_point_empty(geom):
    with pytest.raises(ValueError):
        assert shapely.to_geojson(geom)