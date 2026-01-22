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
def test_to_wkb_byte_order():
    point = shapely.points(1.0, 1.0)
    be = b'\x00'
    le = b'\x01'
    point_type = b'\x01\x00\x00\x00'
    coord = b'\x00\x00\x00\x00\x00\x00\xf0?'
    assert shapely.to_wkb(point, byte_order=1) == le + point_type + 2 * coord
    assert shapely.to_wkb(point, byte_order=0) == be + point_type[::-1] + 2 * coord[::-1]