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
@pytest.mark.parametrize('wkb,expected_type,expected_dim', [(POINT_NAN_WKB, 0, 2), (POINTZ_NAN_WKB, 0, 3), (MULTIPOINT_NAN_WKB, 4, 2), (MULTIPOINTZ_NAN_WKB, 4, 3), (GEOMETRYCOLLECTION_NAN_WKB, 7, 2), (GEOMETRYCOLLECTIONZ_NAN_WKB, 7, 3), (NESTED_COLLECTION_NAN_WKB, 7, 2), (NESTED_COLLECTIONZ_NAN_WKB, 7, 3)])
def test_from_wkb_point_empty(wkb, expected_type, expected_dim):
    geom = shapely.from_wkb(wkb)
    assert shapely.is_empty(geom)
    assert shapely.get_type_id(geom) == expected_type
    if shapely.geos_version >= (3, 9, 0):
        assert shapely.get_coordinate_dimension(geom) == expected_dim