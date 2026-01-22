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
@pytest.mark.parametrize('geometry,exclusive,expected', [(Point(1, 1), False, [1]), ([Point(1, 1)], False, [[0], [1]]), (Point(1, 1), True, [0, 2]), ([Point(1, 1)], True, [[0, 0], [0, 2]]), ([Point(1, 1), Point(2, 2)], True, [[0, 0, 1, 1], [0, 2, 1, 3]])])
def test_query_nearest_exclusive(tree, geometry, exclusive, expected):
    assert_array_equal(tree.query_nearest(geometry, exclusive=exclusive), expected)