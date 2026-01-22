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
def test_different_geo_colname(self):
    data = {'A': range(5), 'B': range(-5, 0), 'location': [Point(x, y) for x, y in zip(range(5), range(5))]}
    df = GeoDataFrame(data, crs=self.crs, geometry='location')
    locs = GeoSeries(data['location'], crs=self.crs)
    assert_geoseries_equal(df.geometry, locs)
    assert 'geometry' not in df
    assert df.geometry.name == 'location'
    assert df._geometry_column_name == 'location'
    geom2 = [Point(x, y) for x, y in zip(range(5, 10), range(5))]
    with pytest.raises(CRSError):
        df.set_geometry(geom2, crs='dummy_crs')