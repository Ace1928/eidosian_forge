import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_shared_paths_linestring():
    g1 = shapely.linestrings([(0, 0), (1, 0), (1, 1)])
    g2 = shapely.linestrings([(0, 0), (1, 0)])
    actual1 = shapely.shared_paths(g1, g2)
    assert_geometries_equal(shapely.get_geometry(actual1, 0), shapely.multilinestrings([g2]))