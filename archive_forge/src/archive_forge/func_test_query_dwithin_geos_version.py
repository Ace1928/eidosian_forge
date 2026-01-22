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
@pytest.mark.skipif(geos_version >= (3, 10, 0), reason='GEOS >= 3.10')
@pytest.mark.parametrize('geometry', [Point(0, 0), [Point(0, 0)], None, [None], empty, [empty]])
def test_query_dwithin_geos_version(tree, geometry):
    with pytest.raises(UnsupportedGEOSVersionError, match='requires GEOS >= 3.10'):
        tree.query(geometry, predicate='dwithin', distance=1)