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
def test_geometry_property_errors(self):
    with pytest.raises(AttributeError):
        df = self.df.copy()
        del df['geometry']
        df.geometry
    with pytest.raises(ValueError):
        df = self.df2.copy()
        df.geometry = 'value1'
    with pytest.raises(ValueError):
        df = self.df.copy()
        df.geometry = 'apple'
    with pytest.raises(TypeError):
        df = self.df.copy()
        df.geometry = list(range(df.shape[0]))
    with pytest.raises(KeyError):
        df = self.df.copy()
        del df['geometry']
        df['geometry']
    with pytest.raises(ValueError):
        df = self.df.copy()
        df.geometry = df