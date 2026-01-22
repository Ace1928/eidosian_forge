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
def test_geo_setitem(self):
    data = {'A': range(5), 'B': np.arange(5.0), 'geometry': [Point(x, y) for x, y in zip(range(5), range(5))]}
    df = GeoDataFrame(data)
    s = GeoSeries([Point(x, y + 1) for x, y in zip(range(5), range(5))])
    for vals in [s, s.values]:
        df['geometry'] = vals
        assert_geoseries_equal(df['geometry'], s)
        assert_geoseries_equal(df.geometry, s)
    s2 = GeoSeries([Point(x, y + 1) for x, y in zip(range(6), range(6))])
    df['geometry'] = s2
    assert_geoseries_equal(df['geometry'], s)
    assert_geoseries_equal(df.geometry, s)
    for vals in [s, s.values]:
        df['other_geom'] = vals
        assert isinstance(df['other_geom'].values, GeometryArray)
    data = {'A': range(5), 'B': np.arange(5.0), 'other_geom': range(5), 'geometry': [Point(x, y) for x, y in zip(range(5), range(5))]}
    df = GeoDataFrame(data)
    for vals in [s, s.values]:
        df['other_geom'] = vals
        assert isinstance(df['other_geom'].values, GeometryArray)