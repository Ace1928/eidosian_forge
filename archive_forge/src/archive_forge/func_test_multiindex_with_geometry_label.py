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
@pytest.mark.parametrize('dtype', ['geometry', 'object'])
def test_multiindex_with_geometry_label(self, dtype):
    df = pd.DataFrame([[Point(0, 0), Point(1, 1)], [Point(2, 2), Point(3, 3)]])
    df = df.astype(dtype)
    df.columns = pd.MultiIndex.from_product([['geometry'], [0, 1]])
    gdf = GeoDataFrame(df)
    with pytest.raises(AttributeError, match='.*geometry .* has not been set.*'):
        gdf.geometry
    res_gdf = gdf.set_geometry(('geometry', 0))
    assert res_gdf.shape == gdf.shape
    assert isinstance(res_gdf.geometry, GeoSeries)