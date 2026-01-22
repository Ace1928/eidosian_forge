import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.skipif(shapely.geos_version < (3, 11, 0), reason='GEOS < 3.11')
def test_concave_hull_kwargs():
    p = Point(10, 10)
    mp = MultiPoint(p.buffer(5).exterior.coords[:] + p.buffer(4).exterior.coords[:])
    result1 = shapely.concave_hull(mp, ratio=0.5)
    assert len(result1.interiors) == 0
    result2 = shapely.concave_hull(mp, ratio=0.5, allow_holes=True)
    assert len(result2.interiors) == 1
    result3 = shapely.concave_hull(mp, ratio=0)
    result4 = shapely.concave_hull(mp, ratio=1)
    assert shapely.get_num_coordinates(result4) < shapely.get_num_coordinates(result3)