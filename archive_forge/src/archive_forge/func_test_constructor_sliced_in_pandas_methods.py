import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
def test_constructor_sliced_in_pandas_methods(df2):
    assert type(df2.count()) == pd.Series
    hashable_test_df = df2.drop(columns=['geometry2', 'geometry3'])
    assert type(hashable_test_df.duplicated()) == pd.Series
    assert type(df2.quantile(numeric_only=True)) == pd.Series
    assert type(df2.memory_usage()) == pd.Series