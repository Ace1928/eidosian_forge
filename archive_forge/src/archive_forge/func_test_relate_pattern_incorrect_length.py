from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
def test_relate_pattern_incorrect_length():
    with pytest.raises(shapely.GEOSException, match='Should be length 9'):
        shapely.relate_pattern(point, polygon, '**')
    with pytest.raises(shapely.GEOSException, match='Should be length 9'):
        shapely.relate_pattern(point, polygon, '**********')