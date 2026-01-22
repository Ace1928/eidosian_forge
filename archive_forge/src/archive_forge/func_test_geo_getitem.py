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
@pytest.mark.filterwarnings('ignore:Geometry is in a geographic CRS')
def test_geo_getitem(self):
    data = {'A': range(5), 'B': range(-5, 0), 'location': [Point(x, y) for x, y in zip(range(5), range(5))]}
    df = GeoDataFrame(data, crs=self.crs, geometry='location')
    assert isinstance(df.geometry, GeoSeries)
    df['geometry'] = df['A']
    assert isinstance(df.geometry, GeoSeries)
    assert df.geometry[0] == data['location'][0]
    assert not isinstance(df['geometry'], GeoSeries)
    assert isinstance(df['location'], GeoSeries)
    df['buff'] = df.buffer(1)
    assert isinstance(df['buff'], GeoSeries)
    df['array'] = from_shapely([Point(x, y) for x, y in zip(range(5), range(5))])
    assert isinstance(df['array'], GeoSeries)
    data['geometry'] = [Point(x + 1, y - 1) for x, y in zip(range(5), range(5))]
    df = GeoDataFrame(data, crs=self.crs)
    assert isinstance(df.geometry, GeoSeries)
    assert isinstance(df['geometry'], GeoSeries)
    assert not isinstance(df['location'], GeoSeries)