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
@pytest.mark.filterwarnings('ignore:Geometry column does not contain geometry:UserWarning')
def test_to_json_geom_col(self):
    df = self.df.copy()
    df['geom'] = df['geometry']
    df['geometry'] = np.arange(len(df))
    df.set_geometry('geom', inplace=True)
    text = df.to_json()
    data = json.loads(text)
    assert data['type'] == 'FeatureCollection'
    assert len(data['features']) == 5