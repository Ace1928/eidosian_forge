import numpy as np
import pytest
from shapely.geometry import Point, Polygon
from shapely.prepared import prep, PreparedGeometry
def test_prepare_already_prepared():
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    prepared = prep(polygon)
    result = prep(prepared)
    assert isinstance(result, PreparedGeometry)
    assert result.context is polygon
    result = PreparedGeometry(prepared)
    assert isinstance(result, PreparedGeometry)
    assert result.context is polygon