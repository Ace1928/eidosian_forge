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
@pytest.mark.skipif(not compat.USE_SHAPELY_20, reason='shapely >= 2.0 is required to run sjoin_nearestwith parameter `exclusive` set')
@pytest.mark.parametrize('max_distance,expected', [(None, expected_index_uncapped), (1.1, [3, 3, 1, 2])])
def test_sjoin_nearest_exclusive(self, max_distance, expected):
    geoms = shapely.points(np.arange(3), np.arange(3))
    geoms = np.append(geoms, [Point(1, 2)])
    df = geopandas.GeoDataFrame({'geometry': geoms})
    result = df.sjoin_nearest(df, max_distance=max_distance, distance_col='dist', exclusive=True)
    assert_series_equal(result['index_right'].reset_index(drop=True), pd.Series(expected), check_names=False)
    if max_distance:
        assert result['dist'].max() <= max_distance