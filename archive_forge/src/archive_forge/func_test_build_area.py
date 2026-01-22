import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 8, 0), reason='GEOS < 3.8')
@pytest.mark.parametrize('geom,expected', [(point, empty), (line_string, empty), (GeometryCollection([Polygon([(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)]), Polygon([(1, 1), (2, 2), (1, 2), (1, 1)])]), Polygon([(0, 0), (0, 3), (3, 3), (3, 0), (0, 0)], holes=[[(1, 1), (2, 2), (1, 2), (1, 1)]])), (empty, empty), ([empty], [empty])])
def test_build_area(geom, expected):
    actual = shapely.build_area(geom)
    assert actual is not expected
    assert actual == expected