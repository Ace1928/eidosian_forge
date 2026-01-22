import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_points_from_coords():
    actual = shapely.points([[0, 0], [2, 2]])
    assert_geometries_equal(actual, [shapely.Point(0, 0), shapely.Point(2, 2)])