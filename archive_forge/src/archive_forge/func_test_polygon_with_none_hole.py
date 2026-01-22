import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygon_with_none_hole():
    actual = shapely.polygons(shapely.linearrings(box_tpl(0, 0, 10, 10)), [shapely.linearrings(box_tpl(1, 1, 2, 2)), None, shapely.linearrings(box_tpl(3, 3, 4, 4))])
    assert shapely.area(actual) == 98.0