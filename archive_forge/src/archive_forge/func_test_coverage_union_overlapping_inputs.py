import numpy as np
import pytest
import shapely
from shapely import Geometry, GeometryCollection, Polygon
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
def test_coverage_union_overlapping_inputs():
    polygon = Polygon([(1, 1), (1, 0), (0, 0), (0, 1), (1, 1)])
    other = Polygon([(1, 0), (0.9, 1), (2, 1), (2, 0), (1, 0)])
    if shapely.geos_version >= (3, 12, 0):
        result = shapely.coverage_union(polygon, other)
        expected = shapely.multipolygons([polygon, other])
        assert_geometries_equal(result, expected, normalize=True)
    else:
        with pytest.raises(shapely.GEOSException, match='CoverageUnion cannot process incorrectly noded inputs.'):
            shapely.coverage_union(polygon, other)