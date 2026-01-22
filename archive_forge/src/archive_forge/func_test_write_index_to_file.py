import datetime
import io
import os
import pathlib
import tempfile
from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
import pytz
from packaging.version import Version
from pandas.api.types import is_datetime64_any_dtype
from pandas.testing import assert_series_equal
from shapely.geometry import Point, Polygon, box
import geopandas
from geopandas import GeoDataFrame, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.io.file import _detect_driver, _EXTENSION_TO_DRIVER
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from geopandas.tests.util import PACKAGE_DIR, validate_boro_df
@pytest.mark.parametrize('driver,ext', [('ESRI Shapefile', 'shp'), ('GeoJSON', 'geojson')])
def test_write_index_to_file(tmpdir, df_points, driver, ext, engine):
    fngen = FileNumber(tmpdir, 'check', ext)

    def do_checks(df, index_is_used):
        other_cols = list(df.columns)
        other_cols.remove('geometry')
        if driver == 'ESRI Shapefile':
            driver_col = ['FID']
        else:
            driver_col = []
        if index_is_used:
            index_cols = list(df.index.names)
        else:
            index_cols = [None] * len(df.index.names)
        if index_cols == [None]:
            index_cols = ['index']
        elif len(index_cols) > 1 and (not all(index_cols)):
            for level, index_col in enumerate(index_cols):
                if index_col is None:
                    index_cols[level] = 'level_' + str(level)
        tempfilename = next(fngen)
        df.to_file(tempfilename, driver=driver, index=None, engine=engine)
        df_check = read_file(tempfilename, engine=engine)
        if len(other_cols) == 0:
            expected_cols = driver_col[:]
        else:
            expected_cols = []
        if index_is_used:
            expected_cols += index_cols
        expected_cols += other_cols + ['geometry']
        assert list(df_check.columns) == expected_cols
        tempfilename = next(fngen)
        df.geometry.to_file(tempfilename, driver=driver, index=None, engine=engine)
        df_check = read_file(tempfilename, engine=engine)
        if index_is_used:
            expected_cols = index_cols + ['geometry']
        else:
            expected_cols = driver_col + ['geometry']
        assert list(df_check.columns) == expected_cols
        tempfilename = next(fngen)
        df.to_file(tempfilename, driver=driver, index=True, engine=engine)
        df_check = read_file(tempfilename, engine=engine)
        assert list(df_check.columns) == index_cols + other_cols + ['geometry']
        tempfilename = next(fngen)
        df.geometry.to_file(tempfilename, driver=driver, index=True, engine=engine)
        df_check = read_file(tempfilename, engine=engine)
        assert list(df_check.columns) == index_cols + ['geometry']
        tempfilename = next(fngen)
        df.to_file(tempfilename, driver=driver, index=False, engine=engine)
        df_check = read_file(tempfilename, engine=engine)
        if len(other_cols) == 0:
            expected_cols = driver_col + ['geometry']
        else:
            expected_cols = other_cols + ['geometry']
        assert list(df_check.columns) == expected_cols
        tempfilename = next(fngen)
        df.geometry.to_file(tempfilename, driver=driver, index=False, engine=engine)
        df_check = read_file(tempfilename, engine=engine)
        assert list(df_check.columns) == driver_col + ['geometry']
    df_p = df_points.copy()
    df = GeoDataFrame(df_p['value1'], geometry=df_p.geometry)
    do_checks(df, index_is_used=False)
    df.index += 1
    do_checks(df, index_is_used=False)
    df_p.index = list(range(1, len(df) + 1))
    df = GeoDataFrame(df_p['value1'], geometry=df_p.geometry)
    do_checks(df, index_is_used=False)
    df_p = df_points.copy()
    df = GeoDataFrame(df_p['value1'], geometry=df_p.geometry).drop(5, axis=0)
    do_checks(df, index_is_used=False)
    df = GeoDataFrame(geometry=df_p.geometry)
    do_checks(df, index_is_used=False)
    df_p = df_points.copy()
    df = GeoDataFrame(df_p['value1'], geometry=df_p.geometry)
    df.index.name = 'foo_index'
    do_checks(df, index_is_used=True)
    df.index.name = 'index'
    do_checks(df, index_is_used=True)
    df_p = df_points.copy()
    df_p['value3'] = df_p['value2'] - df_p['value1']
    df_p.set_index(['value1', 'value2'], inplace=True)
    df = GeoDataFrame(df_p, geometry=df_p.geometry)
    do_checks(df, index_is_used=True)
    df.index.names = ['first', None]
    do_checks(df, index_is_used=True)
    df.index.names = [None, None]
    do_checks(df, index_is_used=True)
    df_p = df_points.copy()
    df = GeoDataFrame(df_p['value1'], geometry=df_p.geometry)
    df.index = df_p.index.astype(float) / 10
    do_checks(df, index_is_used=True)
    df.index.name = 'centile'
    do_checks(df, index_is_used=True)
    df_p = df_points.copy()
    df = GeoDataFrame(df_p['value1'], geometry=df_p.geometry)
    df.index = pd.TimedeltaIndex(range(len(df)), 'days')
    df.index = df.index.astype(str)
    do_checks(df, index_is_used=True)
    df_p = df_points.copy()
    df = GeoDataFrame(df_p['value1'], geometry=df_p.geometry)
    df.index = pd.TimedeltaIndex(range(len(df)), 'days') + pd.DatetimeIndex(['1999-12-27'] * len(df))
    if driver == 'ESRI Shapefile':
        df.index = df.index.astype(str)
    do_checks(df, index_is_used=True)
    df.index.name = 'datetime'
    do_checks(df, index_is_used=True)