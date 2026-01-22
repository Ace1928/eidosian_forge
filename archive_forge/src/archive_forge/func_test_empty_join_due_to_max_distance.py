import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('how', ['inner', 'left'])
def test_empty_join_due_to_max_distance(self, how):
    left = geopandas.GeoDataFrame({'geometry': [Point(0, 0)]})
    right = geopandas.GeoDataFrame({'geometry': [Point(1, 1), Point(2, 2)]})
    joined = sjoin_nearest(left, right, how=how, max_distance=1, distance_col='distances')
    expected = left.copy()
    expected['index_right'] = [np.nan]
    expected['distances'] = [np.nan]
    if how == 'inner':
        expected = expected.dropna()
        expected['index_right'] = expected['index_right'].astype('int64')
    assert_geodataframe_equal(joined, expected)