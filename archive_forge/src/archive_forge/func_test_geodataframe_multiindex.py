import warnings
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_geodataframe_multiindex():

    def create_dataframe():
        gdf = DataFrame([[Point(0, 0), Point(1, 1)], [Point(2, 2), Point(3, 3)]])
        gdf = GeoDataFrame(gdf.astype('geometry'))
        gdf.columns = pd.MultiIndex.from_product([['geometry'], [0, 1]])
        return gdf
    df1 = create_dataframe()
    df2 = create_dataframe()
    assert_geodataframe_equal(df1, df2)
    df1 = create_dataframe()
    df1._geometry_column_name = None
    df2 = create_dataframe()
    df2._geometry_column_name = None
    assert_geodataframe_equal(df1, df2)