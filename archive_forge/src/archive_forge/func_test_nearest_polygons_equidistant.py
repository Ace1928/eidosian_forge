import itertools
import math
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from numpy.testing import assert_array_equal
import shapely
from shapely import box, geos_version, LineString, MultiPoint, Point, STRtree
from shapely.errors import UnsupportedGEOSVersionError
from shapely.testing import assert_geometries_equal
from shapely.tests.common import (
import pickle
import sys
from shapely import Point, geos_version
@pytest.mark.parametrize('geometry,expected', [(Point(0.5, 0.5), [0, 1]), (box(0, 0, 1, 1), [0, 1]), (box(0.5, 0.5, 1.5, 1.5), [0, 1, 2]), (box(3, 3, 5, 5), [3, 4, 5]), (shapely.buffer(Point(2.5, 2.5), HALF_UNIT_DIAG), [2, 3]), (shapely.buffer(Point(3, 3), HALF_UNIT_DIAG), [2, 3, 4]), (MultiPoint([[5, 5], [7, 7]]), [5, 7]), (MultiPoint([[5.5, 5], [7, 7]]), [5, 7])])
def test_nearest_polygons_equidistant(poly_tree, geometry, expected):
    result = poly_tree.nearest(geometry)
    assert result in expected