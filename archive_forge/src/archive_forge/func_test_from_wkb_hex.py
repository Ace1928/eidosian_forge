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
def test_from_wkb_hex():
    expected = shapely.points(1, 1)
    actual = shapely.from_wkb('0101000000000000000000F03F000000000000F03F')
    assert_geometries_equal(actual, expected)
    actual = shapely.from_wkb(b'0101000000000000000000F03F000000000000F03F')
    assert_geometries_equal(actual, expected)