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
@pytest.mark.parametrize('predicate', ['bad_predicate', 'disjoint'])
def test_query_invalid_predicate(tree, predicate):
    with pytest.raises(ValueError, match='is not a valid option'):
        tree.query(Point(1, 1), predicate=predicate)