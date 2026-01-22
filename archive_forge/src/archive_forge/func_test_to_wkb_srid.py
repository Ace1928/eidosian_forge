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
def test_to_wkb_srid():
    ewkb = '01010000200400000000000000000000000000000000000000'
    wkb = '010100000000000000000000000000000000000000'
    actual = shapely.from_wkb(ewkb)
    assert shapely.to_wkt(actual, trim=True) == 'POINT (0 0)'
    assert shapely.to_wkb(actual, hex=True, byte_order=1) == wkb
    assert shapely.to_wkb(actual, hex=True, include_srid=True, byte_order=1) == ewkb
    point = shapely.points(1, 1)
    point_with_srid = shapely.set_srid(point, np.int32(4326))
    result = shapely.to_wkb(point_with_srid, include_srid=True, byte_order=1)
    assert np.frombuffer(result[5:9], '<u4').item() == 4326