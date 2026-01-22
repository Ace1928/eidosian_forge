import warnings
import numpy as np
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_no_crs():
    df1 = GeoDataFrame({'col1': [1, 2], 'geometry': s1}, crs=None)
    df2 = GeoDataFrame({'col1': [1, 2], 'geometry': s1}, crs={})
    assert_geodataframe_equal(df1, df2)