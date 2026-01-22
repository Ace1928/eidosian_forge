import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason='GEOS < 3.7')
@pytest.mark.parametrize('geom1, geom2, densify, expected', [(shapely.linestrings([[0, 0], [100, 0]]), shapely.linestrings([[0, 0], [50, 50], [100, 0]]), 0.001, 50)])
def test_frechet_distance_densify(geom1, geom2, densify, expected):
    actual = shapely.frechet_distance(geom1, geom2, densify=densify)
    assert actual == pytest.approx(expected, abs=1e-12)