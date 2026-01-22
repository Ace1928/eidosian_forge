import platform
import weakref
import numpy as np
import pytest
import shapely
from shapely import (
from shapely.errors import ShapelyDeprecationWarning
from shapely.testing import assert_geometries_equal
@pytest.mark.skipif(shapely.geos_version < (3, 10, 0), reason='GEOS < 3.10')
def test_dwithin():
    point = Point(1, 1)
    line = LineString([(0, 0), (0, 10)])
    assert point.dwithin(line, 0.5) is False
    assert point.dwithin(line, 1.5) is True