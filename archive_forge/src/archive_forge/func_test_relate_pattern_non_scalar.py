from functools import partial
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, Point
from shapely.tests.common import (
def test_relate_pattern_non_scalar():
    with pytest.raises(ValueError, match='only supports scalar'):
        shapely.relate_pattern([point] * 2, polygon, ['*********'] * 2)