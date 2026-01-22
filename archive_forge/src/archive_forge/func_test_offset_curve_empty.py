import numpy as np
import pytest
import shapely
from shapely import (
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
def test_offset_curve_empty():
    with ignore_invalid(shapely.geos_version < (3, 12, 0)):
        actual = shapely.offset_curve(empty_line_string, 2.0)
    assert shapely.is_empty(actual)