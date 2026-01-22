import warnings
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_geodataframe():
    assert_geodataframe_equal(df1, df2)
    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2, check_less_precise=True)
    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df2[['geometry', 'col1']])
    assert_geodataframe_equal(df1, df2[['geometry', 'col1']], check_like=True)
    df3 = df2.copy()
    df3.loc[0, 'col1'] = 10
    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df1, df3)
    assert_geodataframe_equal(df5, df4, check_like=True)
    df5.geom2.crs = 3857
    with pytest.raises(AssertionError):
        assert_geodataframe_equal(df5, df4, check_like=True)