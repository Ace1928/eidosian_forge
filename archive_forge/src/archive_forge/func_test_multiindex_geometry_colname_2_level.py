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
def test_multiindex_geometry_colname_2_level(self):
    crs = 'EPSG:4326'
    df = pd.DataFrame([[1, 0], [0, 1]], columns=[['location', 'location'], ['x', 'y']])
    x_col = df['location', 'x']
    y_col = df['location', 'y']
    gdf = GeoDataFrame(df, crs=crs, geometry=points_from_xy(x_col, y_col))
    assert gdf.crs == crs
    assert gdf.geometry.crs == crs
    assert gdf.geometry.dtype == 'geometry'
    assert gdf._geometry_column_name == 'geometry'
    assert gdf.geometry.name == 'geometry'