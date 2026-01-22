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
def test_from_wkb_ignore_on_invalid_ignore():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = shapely.from_wkb(b'\x01\x01\x00\x00\x00\x00', on_invalid='ignore')
        assert result is None
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = shapely.from_wkb(INVALID_WKB, on_invalid='ignore')
        assert result is None