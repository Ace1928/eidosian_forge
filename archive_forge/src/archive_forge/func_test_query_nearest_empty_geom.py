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
@pytest.mark.parametrize('geometry,expected', [(empty, []), ([empty], [[], []]), ([empty, point], [[1, 1], [2, 3]])])
def test_query_nearest_empty_geom(tree, geometry, expected):
    assert_array_equal(tree.query_nearest(geometry), expected)