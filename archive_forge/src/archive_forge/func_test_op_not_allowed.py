import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.prepared import prep, PreparedGeometry
def test_op_not_allowed():
    p = PreparedGeometry(Point(0.0, 0.0).buffer(1.0))
    with pytest.raises(TypeError):
        Point(0.0, 0.0).union(p)