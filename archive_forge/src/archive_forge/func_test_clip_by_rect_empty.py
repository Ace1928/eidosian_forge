import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('geom', [empty, empty_line_string, empty_polygon])
def test_clip_by_rect_empty(geom):
    actual = shapely.clip_by_rect(geom, 0, 0, 1, 1)
    assert actual == GeometryCollection()