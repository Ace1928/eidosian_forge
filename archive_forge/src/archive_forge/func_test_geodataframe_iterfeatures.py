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
def test_geodataframe_iterfeatures(self):
    df = self.df.iloc[:1].copy()
    df.loc[0, 'BoroName'] = np.nan
    result = list(df.iterfeatures(na='null'))[0]['properties']
    assert result['BoroName'] is None
    result = list(df.iterfeatures(na='drop'))[0]['properties']
    assert 'BoroName' not in result.keys()
    result = list(df.iterfeatures(na='keep'))[0]['properties']
    assert np.isnan(result['BoroName'])
    assert type(df.loc[0, 'Shape_Leng']) is np.float64
    result = list(df.iterfeatures(na='null'))[0]
    assert type(result['properties']['Shape_Leng']) is float
    result = list(df.iterfeatures(na='drop'))[0]
    assert type(result['properties']['Shape_Leng']) is float
    result = list(df.iterfeatures(na='keep'))[0]
    assert type(result['properties']['Shape_Leng']) is float
    df_only_numerical_cols = df[['Shape_Leng', 'Shape_Area', 'geometry']]
    assert type(df_only_numerical_cols.loc[0, 'Shape_Leng']) is np.float64
    result = list(df_only_numerical_cols.iterfeatures(na='null'))[0]
    assert type(result['properties']['Shape_Leng']) is float
    result = list(df_only_numerical_cols.iterfeatures(na='drop'))[0]
    assert type(result['properties']['Shape_Leng']) is float
    result = list(df_only_numerical_cols.iterfeatures(na='keep'))[0]
    assert type(result['properties']['Shape_Leng']) is float
    with pytest.raises(ValueError, match='GeoDataFrame cannot contain duplicated column names.'):
        df_with_duplicate_columns = df[['Shape_Leng', 'Shape_Leng', 'Shape_Area', 'geometry']]
        list(df_with_duplicate_columns.iterfeatures())
    df = GeoDataFrame({'values': [0, 1], 'geom': [Point(0, 1), Point(1, 0)]})
    with pytest.raises(AttributeError):
        list(df.iterfeatures())