import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
@pytest.mark.parametrize('geom,shape', [(point, (4,)), (None, (4,)), ([point, multi_point], (2, 4)), ([[point, multi_point], [polygon, point]], (2, 2, 4)), ([[[point, multi_point]], [[polygon, point]]], (2, 1, 2, 4))])
def test_bounds_dimensions(geom, shape):
    assert shapely.bounds(geom).shape == shape