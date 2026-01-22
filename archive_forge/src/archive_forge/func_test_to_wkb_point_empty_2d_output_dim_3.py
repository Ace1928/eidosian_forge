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
@pytest.mark.xfail(shapely.geos_version < (3, 8, 0), reason='GEOS<3.8 always outputs 3D empty points if output_dimension=3')
@pytest.mark.parametrize('geom,expected', [(empty_point, POINT_NAN_WKB), (shapely.multipoints([empty_point]), MULTIPOINT_NAN_WKB), (shapely.geometrycollections([empty_point]), GEOMETRYCOLLECTION_NAN_WKB), (shapely.geometrycollections([shapely.multipoints([empty_point])]), NESTED_COLLECTION_NAN_WKB)])
def test_to_wkb_point_empty_2d_output_dim_3(geom, expected):
    actual = shapely.to_wkb(geom, output_dimension=3, byte_order=1)
    coordinate_length = 16
    header_length = len(expected) - coordinate_length
    assert len(actual) == header_length + coordinate_length
    assert actual[:header_length] == expected[:header_length]
    assert np.isnan(struct.unpack('<2d', actual[header_length:])).all()