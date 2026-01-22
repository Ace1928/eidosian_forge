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
def test_set_geometry_inplace(self):
    geom = [Point(x, y) for x, y in zip(range(5), range(5))]
    ret = self.df.set_geometry(geom, inplace=True)
    assert ret is None
    geom = GeoSeries(geom, index=self.df.index, crs=self.df.crs)
    assert_geoseries_equal(self.df.geometry, geom)