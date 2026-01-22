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
@pytest.mark.parametrize('predicate', ['intersects', 'within', 'contains'])
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20 or compat.HAS_RTREE), reason='sjoin needs `rtree` or `pygeos` dependency')
def test_sjoin(self, how, predicate):
    """
        Basic test for availability of the GeoDataFrame method. Other
        sjoin tests are located in /tools/tests/test_sjoin.py
        """
    left = read_file(geopandas.datasets.get_path('naturalearth_cities'))
    right = read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    expected = geopandas.sjoin(left, right, how=how, predicate=predicate)
    result = left.sjoin(right, how=how, predicate=predicate)
    assert_geodataframe_equal(result, expected)