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
def test_getitem_no_geometry(self):
    res = self.df2[['value1', 'value2']]
    assert isinstance(res, pd.DataFrame)
    assert not isinstance(res, GeoDataFrame)
    df = self.df2.copy()
    df = df.rename(columns={'geometry': 'geom'}).set_geometry('geom')
    assert isinstance(df, GeoDataFrame)
    res = df[['value1', 'value2']]
    assert isinstance(res, pd.DataFrame)
    assert not isinstance(res, GeoDataFrame)
    df['geometry'] = np.arange(len(df))
    res = df[['value1', 'value2', 'geometry']]
    assert isinstance(res, pd.DataFrame)
    assert not isinstance(res, GeoDataFrame)