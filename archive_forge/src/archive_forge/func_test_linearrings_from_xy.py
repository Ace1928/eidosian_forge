import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_linearrings_from_xy():
    actual = shapely.linearrings([0, 1, 2, 0], [3, 4, 5, 3])
    assert_geometries_equal(actual, LinearRing([(0, 3), (1, 4), (2, 5), (0, 3)]))