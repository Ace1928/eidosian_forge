import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason='GEOS < 3.7')
@pytest.mark.parametrize('geom1, geom2, expected', [(shapely.linestrings([[0, 0], [100, 0]]), shapely.linestrings([[0, 0], [100, 0]]), 0), (shapely.linestrings([[0, 0], [50, 200], [100, 0], [150, 200], [200, 0]]), shapely.linestrings([[0, 200], [200, 150], [0, 100], [200, 50], [0, 0]]), 200), (shapely.linestrings([[0, 0], [50, 200], [100, 0], [150, 200], [200, 0]]), shapely.linestrings([[200, 0], [150, 200], [100, 0], [50, 200], [0, 0]]), 200), (shapely.linestrings([[0, 0], [50, 200], [100, 0], [150, 200], [200, 0]]), shapely.linestrings([[0, 0], [200, 50], [0, 100], [200, 150], [0, 200]]), 282.842712474619), (shapely.linestrings([[0, 0], [100, 0]]), shapely.linestrings([[0, 0], [50, 50], [100, 0]]), 70.7106781186548)])
def test_frechet_distance(geom1, geom2, expected):
    actual = shapely.frechet_distance(geom1, geom2)
    assert actual == pytest.approx(expected, abs=1e-12)