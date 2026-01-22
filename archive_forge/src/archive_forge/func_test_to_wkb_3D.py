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
def test_to_wkb_3D():
    point_z = shapely.points(1, 1, 1)
    actual = shapely.to_wkb(point_z, byte_order=1)
    assert actual == b'\x01\x01\x00\x00\x80\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\xf0?'
    actual = shapely.to_wkb(point_z, output_dimension=2, byte_order=1)
    assert actual == POINT11_WKB