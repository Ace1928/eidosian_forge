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
@pytest.mark.parametrize('geometry,exclusive', [(Point(1, 1), 'invalid'), (Point(1, 1), ['also invalid']), ([Point(1, 1)], []), ([Point(1, 1)], [False])])
def test_query_nearest_invalid_exclusive(tree, geometry, exclusive):
    with pytest.raises(ValueError):
        tree.query_nearest(geometry, exclusive=exclusive)