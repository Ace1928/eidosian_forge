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
@pytest.mark.skipif(geos_version < (3, 10, 0), reason='GEOS < 3.10')
@pytest.mark.parametrize('geometry,distance,expected', [(Point(0, 0), 0, [0]), ([Point(0, 0)], 0, [[0], [0]]), (Point(0, 0), 0.5, [0]), ([Point(0, 0)], 0.5, [[0], [0]]), (Point(0, 0), 1.5, [0, 1]), ([Point(0, 0)], 1.5, [[0, 0], [0, 1]]), (Point(0.5, 0.5), 1, [0, 1]), ([Point(0.5, 0.5)], 1, [[0, 0], [0, 1]]), (Point(0.5, 0.5), 0.5, [0, 1]), ([Point(0.5, 0.5)], 0.5, [[0, 0], [0, 1]]), (box(0, 0, 1, 1), 0, [0, 1]), ([box(0, 0, 1, 1)], 0, [[0, 0], [0, 1]]), (box(0, 0, 1, 1), 2, [0, 1, 2]), ([box(0, 0, 1, 1)], 2, [[0, 0, 0], [0, 1, 2]]), (MultiPoint([[5, 5], [7, 7]]), 0.5, [5, 7]), ([MultiPoint([[5, 5], [7, 7]])], 0.5, [[0, 0], [5, 7]]), (MultiPoint([[5, 5], [7, 7]]), 2.5, [3, 4, 5, 6, 7, 8, 9]), ([MultiPoint([[5, 5], [7, 7]])], 2.5, [[0, 0, 0, 0, 0, 0, 0], [3, 4, 5, 6, 7, 8, 9]])])
def test_query_dwithin_polygons(poly_tree, geometry, distance, expected):
    assert_array_equal(poly_tree.query(geometry, predicate='dwithin', distance=distance), expected)