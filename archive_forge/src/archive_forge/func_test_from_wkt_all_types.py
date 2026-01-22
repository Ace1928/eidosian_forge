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
@pytest.mark.parametrize('geom', all_types)
def test_from_wkt_all_types(geom):
    wkt = shapely.to_wkt(geom)
    actual = shapely.from_wkt(wkt)
    assert_geometries_equal(actual, geom)