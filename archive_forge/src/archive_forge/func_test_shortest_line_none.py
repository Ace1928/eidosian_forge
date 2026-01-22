import numpy as np
import pytest
import shapely
from shapely import GeometryCollection, LinearRing, LineString, MultiLineString, Point
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
@pytest.mark.parametrize('prepare', [True, False])
def test_shortest_line_none(prepare):
    assert shapely.shortest_line(_prepare_input(line_string, prepare), None) is None
    assert shapely.shortest_line(None, line_string) is None
    assert shapely.shortest_line(None, None) is None