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
def test_to_wkb_hex():
    point = shapely.points(1, 1)
    actual = shapely.to_wkb(point, hex=True, byte_order=1)
    le = '01'
    point_type = '01000000'
    coord = '000000000000F03F'
    assert actual == le + point_type + 2 * coord