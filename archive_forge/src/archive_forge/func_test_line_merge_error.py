import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version >= (3, 11, 0), reason='GEOS >= 3.11.0')
def test_line_merge_error():
    lines = MultiLineString([[(0, 0), (1, 0)], [(0, 0), (3, 0)]])
    with pytest.raises(UnsupportedGEOSVersionError):
        shapely.line_merge(lines, directed=True)