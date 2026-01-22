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
def test_from_wkb_warn_on_invalid_warn():
    with pytest.warns(Warning, match='Invalid WKB'):
        result = shapely.from_wkb(b'\x01\x01\x00\x00\x00\x00', on_invalid='warn')
        assert result is None
    with pytest.warns(Warning, match='Invalid WKB'):
        result = shapely.from_wkb(INVALID_WKB, on_invalid='warn')
        assert result is None