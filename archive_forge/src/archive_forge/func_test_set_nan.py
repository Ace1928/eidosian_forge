import warnings
import numpy as np
import pytest
import shapely
from shapely import LinearRing, LineString, MultiPolygon, Point, Polygon
from shapely.testing import assert_geometries_equal
from shapely.tests.common import all_types
from shapely.tests.common import empty as empty_geometry_collection
from shapely.tests.common import (
def test_set_nan():
    with ignore_invalid():
        a = set(shapely.linestrings([[[np.nan, np.nan], [np.nan, np.nan]]] * 10))
    assert len(a) == 10