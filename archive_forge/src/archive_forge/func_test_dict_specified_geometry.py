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
def test_dict_specified_geometry(self):
    data = {'A': range(3), 'B': np.arange(3.0), 'other_geom': [Point(x, x) for x in range(3)]}
    df = GeoDataFrame(data, geometry='other_geom')
    check_geodataframe(df, 'other_geom')
    with pytest.raises(ValueError):
        df = GeoDataFrame(data, geometry='geometry')
    df = GeoDataFrame(data)
    with pytest.raises(AttributeError):
        _ = df.geometry
    df = df.set_geometry('other_geom')
    check_geodataframe(df, 'other_geom')
    df = GeoDataFrame(data, geometry='other_geom', columns=['B', 'other_geom'])
    check_geodataframe(df, 'other_geom')
    assert_index_equal(df.columns, pd.Index(['B', 'other_geom']))
    assert_series_equal(df['B'], pd.Series(np.arange(3.0), name='B'))
    df = GeoDataFrame(data, geometry='other_geom', columns=['other_geom', 'A'])
    check_geodataframe(df, 'other_geom')
    assert_index_equal(df.columns, pd.Index(['other_geom', 'A']))
    assert_series_equal(df['A'], pd.Series(range(3), name='A'))