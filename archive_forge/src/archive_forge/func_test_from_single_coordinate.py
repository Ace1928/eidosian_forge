import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.coords import CoordinateSequence
def test_from_single_coordinate():
    """Test for issue #486"""
    coords = [[-122.185933073564, 37.3629353839073]]
    with pytest.raises(shapely.GEOSException):
        ls = LineString(coords)
        ls.geom_type