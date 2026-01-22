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
def test_set_geometry_series(self):
    self.df.index = range(len(self.df) - 1, -1, -1)
    d = {}
    for i in range(len(self.df)):
        d[i] = Point(i, i)
    g = GeoSeries(d)
    df = self.df.set_geometry(g)
    for i, r in df.iterrows():
        assert i == r['geometry'].x
        assert i == r['geometry'].y