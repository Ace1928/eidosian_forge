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
def test_set_geometry(self):
    geom = GeoSeries([Point(x, y) for x, y in zip(range(5), range(5))])
    original_geom = self.df.geometry
    df2 = self.df.set_geometry(geom)
    assert self.df is not df2
    assert_geoseries_equal(df2.geometry, geom, check_crs=False)
    assert_geoseries_equal(self.df.geometry, original_geom)
    assert_geoseries_equal(self.df['geometry'], self.df.geometry)
    with pytest.raises(ValueError):
        self.df.set_geometry('nonexistent-column')
    with pytest.raises(ValueError):
        self.df.set_geometry(self.df)
    gs = GeoSeries(geom, crs='epsg:3857')
    new_df = self.df.set_geometry(gs)
    assert new_df.crs == 'epsg:3857'
    new_df = self.df.set_geometry(gs, crs='epsg:26909')
    assert new_df.crs == 'epsg:26909'
    assert new_df.geometry.crs == 'epsg:26909'
    new_df = self.df.set_geometry(geom.values)
    assert new_df.crs == self.df.crs
    assert new_df.geometry.crs == self.df.crs