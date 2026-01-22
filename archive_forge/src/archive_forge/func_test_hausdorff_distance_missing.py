import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
def test_hausdorff_distance_missing():
    actual = shapely.hausdorff_distance(point, None)
    assert np.isnan(actual)
    actual = shapely.hausdorff_distance(point, None, densify=0.001)
    assert np.isnan(actual)