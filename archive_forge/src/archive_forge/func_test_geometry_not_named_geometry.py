import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
@pytest.mark.parametrize('other_geometry', [False, True])
def test_geometry_not_named_geometry(dfs, how, other_geometry):
    df1, df2 = dfs
    df3 = df1.copy()
    df3 = df3.rename(columns={'geometry': 'polygons'})
    df3 = df3.set_geometry('polygons')
    if other_geometry:
        df3['geometry'] = df1.centroid.geometry
    assert df3.geometry.name == 'polygons'
    res1 = overlay(df1, df2, how=how)
    res2 = overlay(df3, df2, how=how)
    assert df3.geometry.name == 'polygons'
    if how == 'difference':
        assert res2.geometry.name == 'polygons'
        if other_geometry:
            assert 'geometry' in res2.columns
            assert_geoseries_equal(res2['geometry'], df3['geometry'], check_series_type=False)
            res2 = res2.drop(['geometry'], axis=1)
        res2 = res2.rename(columns={'polygons': 'geometry'})
        res2 = res2.set_geometry('geometry')
    if other_geometry and how == 'intersection':
        res2 = res2.reindex(columns=res1.columns)
    assert_geodataframe_equal(res1, res2)
    df4 = df2.copy()
    df4 = df4.rename(columns={'geometry': 'geom'})
    df4 = df4.set_geometry('geom')
    if other_geometry:
        df4['geometry'] = df2.centroid.geometry
    assert df4.geometry.name == 'geom'
    res1 = overlay(df1, df2, how=how)
    res2 = overlay(df1, df4, how=how)
    assert_geodataframe_equal(res1, res2)