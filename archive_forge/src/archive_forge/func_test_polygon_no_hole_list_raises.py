import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_polygon_no_hole_list_raises():
    with pytest.raises(ValueError):
        shapely.polygons(box_tpl(0, 0, 10, 10), box_tpl(1, 1, 2, 2))