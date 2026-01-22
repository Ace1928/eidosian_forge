import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 7, 0), reason='GEOS < 3.7')
@pytest.mark.parametrize('geom1, geom2', [(line_string, None), (None, line_string), (None, None), (line_string, empty), (empty, line_string), (empty, empty)])
def test_frechet_distance_nan_for_invalid_geometry_inputs(geom1, geom2):
    actual = shapely.frechet_distance(geom1, geom2)
    assert np.isnan(actual)