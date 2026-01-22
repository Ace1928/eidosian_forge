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
def test_from_frame_specified_geometry(self):
    data = {'A': range(3), 'B': np.arange(3.0), 'other_geom': [Point(x, x) for x in range(3)]}
    gpdf = GeoDataFrame(data, geometry='other_geom')
    check_geodataframe(gpdf, 'other_geom')
    with ignore_shapely2_warnings():
        pddf = pd.DataFrame(data)
    for df in [gpdf, pddf]:
        res = GeoDataFrame(df, geometry='other_geom')
        check_geodataframe(res, 'other_geom')
    df = GeoDataFrame(gpdf)
    check_geodataframe(df, 'other_geom')