from __future__ import absolute_import
from itertools import product
import json
from packaging.version import Version
import os
import pathlib
import pytest
from pandas import DataFrame, read_parquet as pd_read_parquet
from pandas.testing import assert_frame_equal
import numpy as np
import pyproj
import shapely
from shapely.geometry import box, Point, MultiPolygon
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, read_file, read_parquet, read_feather
from geopandas.array import to_wkb
from geopandas.datasets import get_path
from geopandas.io.arrow import (
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import mock
def test_read_gdal_files():
    """
    Verify that files written by GDAL can be read by geopandas.
    Since it is currently not yet straightforward to install GDAL with
    Parquet/Arrow enabled in our conda setup, we are testing with some
    generated files included in the repo (using GDAL 3.5.0):

    # small dummy test dataset (not naturalearth_lowres, as this can change over time)
    from shapely.geometry import box, MultiPolygon
    df = geopandas.GeoDataFrame(
        {"col_str": ["a", "b"], "col_int": [1, 2], "col_float": [0.1, 0.2]},
        geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5,5)],
        crs="EPSG:4326",
    )
    df.to_file("test_data.gpkg", GEOMETRY_NAME="geometry")
    and then the gpkg file is converted to Parquet/Arrow with:
    $ ogr2ogr -f Parquet -lco FID= test_data_gdal350.parquet test_data.gpkg
    $ ogr2ogr -f Arrow -lco FID= -lco GEOMETRY_ENCODING=WKB test_data_gdal350.arrow test_data.gpkg
    """
    expected = geopandas.GeoDataFrame({'col_str': ['a', 'b'], 'col_int': [1, 2], 'col_float': [0.1, 0.2]}, geometry=[MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]), box(4, 4, 5, 5)], crs='EPSG:4326')
    df = geopandas.read_parquet(DATA_PATH / 'arrow' / 'test_data_gdal350.parquet')
    assert_geodataframe_equal(df, expected, check_crs=True)
    df = geopandas.read_feather(DATA_PATH / 'arrow' / 'test_data_gdal350.arrow')
    assert_geodataframe_equal(df, expected, check_crs=True)