from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
def test_relate_pattern_empty():
    with ignore_invalid(shapely.geos_version < (3, 12, 0)):
        assert shapely.relate_pattern(empty, empty, '*' * 9).item() is True