import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import shapely
from shapely import GeometryCollection, LineString, MultiPoint, Point, Polygon
from shapely.tests.common import (
def test_hausdorff_distance_densify_empty():
    actual = shapely.hausdorff_distance(point, empty, densify=0.2)
    assert np.isnan(actual)