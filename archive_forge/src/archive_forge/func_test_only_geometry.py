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
def test_only_geometry(self):
    exp = GeoDataFrame({'geometry': [Point(x, x) for x in range(3)], 'other': range(3)})[['geometry']]
    df = GeoDataFrame(geometry=[Point(x, x) for x in range(3)])
    check_geodataframe(df)
    assert_geodataframe_equal(df, exp)
    df = GeoDataFrame({'geometry': [Point(x, x) for x in range(3)]})
    check_geodataframe(df)
    assert_geodataframe_equal(df, exp)
    df = GeoDataFrame({'other_geom': [Point(x, x) for x in range(3)]}, geometry='other_geom')
    check_geodataframe(df, 'other_geom')
    exp = exp.rename(columns={'geometry': 'other_geom'}).set_geometry('other_geom')
    assert_geodataframe_equal(df, exp)