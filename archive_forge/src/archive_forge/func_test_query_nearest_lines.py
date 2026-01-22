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
@pytest.mark.parametrize('geometry,expected', [(Point(0.5, 0.5), [0]), ([Point(0.5, 0.5)], [[0], [0]]), (Point(2, 2), [1, 2]), ([Point(2, 2)], [[0, 0], [1, 2]]), (box(0, 0, 1, 1), [0, 1]), ([box(0, 0, 1, 1)], [[0, 0], [0, 1]]), (box(0.5, 0.5, 1.5, 1.5), [0, 1]), ([box(0.5, 0.5, 1.5, 1.5)], [[0, 0], [0, 1]]), ([box(0, 0, 0.5, 0.5), box(3, 3, 5, 5)], [[0, 1, 1, 1, 1], [0, 2, 3, 4, 5]]), (shapely.buffer(Point(2.5, 2.5), 1), [1, 2, 3]), ([shapely.buffer(Point(2.5, 2.5), 1)], [[0, 0, 0], [1, 2, 3]]), (shapely.buffer(Point(3, 3), 0.5), [2, 3]), ([shapely.buffer(Point(3, 3), 0.5)], [[0, 0], [2, 3]]), (MultiPoint([[5, 5], [7, 7]]), [4, 5, 6, 7]), ([MultiPoint([[5, 5], [7, 7]])], [[0, 0, 0, 0], [4, 5, 6, 7]]), (MultiPoint([[5.5, 5], [7, 7]]), [6, 7]), ([MultiPoint([[5.5, 5], [7, 7]])], [[0, 0], [6, 7]]), (MultiPoint([[5, 7], [7, 5]]), [5, 6]), ([MultiPoint([[5, 7], [7, 5]])], [[0, 0], [5, 6]])])
def test_query_nearest_lines(line_tree, geometry, expected):
    assert_array_equal(line_tree.query_nearest(geometry), expected)