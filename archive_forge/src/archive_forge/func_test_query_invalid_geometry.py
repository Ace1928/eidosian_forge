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
@pytest.mark.parametrize('geometry', ['I am not a geometry', ['I am not a geometry'], [Point(0, 0), 'still not a geometry'], [[], 'in a mixed array', 1]])
@pytest.mark.filterwarnings('ignore:Creating an ndarray from ragged nested sequences:')
def test_query_invalid_geometry(tree, geometry):
    with pytest.raises((TypeError, ValueError)):
        tree.query(geometry)