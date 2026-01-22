from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skipif(not compat.USE_SHAPELY_20, reason='shapely >= 2.0 is required to test sindex.nearest with parameter exclusive')
@pytest.mark.parametrize('return_distance', [True, False])
@pytest.mark.parametrize('return_all,max_distance,exclusive,expected', [(False, None, False, ([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], 5 * [0])), (False, None, True, ([[0, 1, 2, 3, 4], [1, 0, 1, 2, 3]], 5 * [sqrt(2)])), (True, None, False, ([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], 5 * [0])), (True, None, True, ([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], 8 * [sqrt(2)])), (False, 1.1, True, ([[1, 2, 5], [5, 5, 1]], 3 * [1])), (True, 1.1, True, ([[1, 2, 5, 5], [5, 5, 1, 2]], 4 * [1]))])
def test_nearest_exclusive(self, expected, max_distance, return_all, return_distance, exclusive):
    geoms = mod.points(np.arange(5), np.arange(5))
    if max_distance:
        geoms = np.append(geoms, [Point(1, 2)])
    df = geopandas.GeoDataFrame({'geometry': geoms})
    ps = geoms
    res = df.sindex.nearest(ps, return_all=return_all, max_distance=max_distance, return_distance=return_distance, exclusive=exclusive)
    if return_distance:
        assert_array_equal(res[0], expected[0])
        assert_array_equal(res[1], expected[1])
    else:
        assert_array_equal(res, expected[0])