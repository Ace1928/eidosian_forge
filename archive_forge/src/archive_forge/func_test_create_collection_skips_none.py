import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('func,sub_geom', [(shapely.multipoints, point), (shapely.multilinestrings, line_string), (shapely.multipolygons, polygon), (shapely.geometrycollections, polygon)])
def test_create_collection_skips_none(func, sub_geom):
    actual = func([sub_geom, None, None, sub_geom])
    assert shapely.get_num_geometries(actual) == 2