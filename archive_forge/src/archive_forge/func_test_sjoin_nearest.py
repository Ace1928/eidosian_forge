import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from pyproj import CRS
from pyproj.exceptions import CRSError
from shapely.geometry import Point, Polygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file
from geopandas.array import GeometryArray, GeometryDtype, from_shapely
from geopandas._compat import ignore_shapely2_warnings
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
@pytest.mark.parametrize('how', ['left', 'inner', 'right'])
@pytest.mark.parametrize('max_distance', [None, 1])
@pytest.mark.parametrize('distance_col', [None, 'distance'])
@pytest.mark.skipif(not TEST_NEAREST, reason='PyGEOS >= 0.10.0 must be installed and activated via the geopandas.compat module to test sjoin_nearest')
def test_sjoin_nearest(self, how, max_distance, distance_col):
    """
        Basic test for availability of the GeoDataFrame method. Other
        sjoin tests are located in /tools/tests/test_sjoin.py
        """
    left = read_file(geopandas.datasets.get_path('naturalearth_cities'))
    right = read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    expected = geopandas.sjoin_nearest(left, right, how=how, max_distance=max_distance, distance_col=distance_col)
    result = left.sjoin_nearest(right, how=how, max_distance=max_distance, distance_col=distance_col)
    assert_geodataframe_equal(result, expected)