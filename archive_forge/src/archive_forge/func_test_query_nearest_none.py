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
@pytest.mark.parametrize('geometry,return_distance,expected', [(None, False, []), ([None], False, [[], []]), (None, True, ([], [])), ([None], True, ([[], []], []))])
def test_query_nearest_none(tree, geometry, return_distance, expected):
    if return_distance:
        index, distance = tree.query_nearest(geometry, return_distance=True)
        assert_array_equal(index, expected[0])
        assert_array_equal(distance, expected[1])
    else:
        assert_array_equal(tree.query_nearest(geometry), expected)