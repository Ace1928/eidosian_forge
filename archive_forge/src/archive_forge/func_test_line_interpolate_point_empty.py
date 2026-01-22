import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('geom', [LineString(), LinearRing(), MultiLineString(), shapely.from_wkt('MULTILINESTRING (EMPTY, (0 0, 1 1))'), GeometryCollection(), GeometryCollection([LineString(), Point(1, 1)])])
def test_line_interpolate_point_empty(geom, normalized):
    assert_geometries_equal(shapely.line_interpolate_point(geom, 0.2, normalized=normalized), empty_point)