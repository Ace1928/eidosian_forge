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
@pytest.mark.parametrize('geometry,expected', [(Point(0.5, 0.5), [0, 1]), (box(0, 0, 3, 3), [0, 1, 2, 3]), (MultiPoint([[5, 5], [7, 7]]), [5, 7])])
def test_nearest_points_equidistant(tree, geometry, expected):
    result = tree.nearest(geometry)
    assert result in expected