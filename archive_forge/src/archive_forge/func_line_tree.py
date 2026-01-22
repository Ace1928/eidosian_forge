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
@pytest.fixture(scope='session')
def line_tree():
    x = np.arange(10)
    y = np.arange(10)
    offset = 1
    geoms = shapely.linestrings(np.array([[x, x + offset], [y, y + offset]]).T)
    yield STRtree(geoms)