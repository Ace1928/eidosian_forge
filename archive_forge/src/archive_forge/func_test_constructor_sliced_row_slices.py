import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
@pytest.mark.parametrize('column_set', test_case_column_sets, ids=[', '.join(i) for i in test_case_column_sets])
def test_constructor_sliced_row_slices(df2, column_set):
    df_subset = df2[column_set]
    assert isinstance(df_subset, GeoDataFrame)
    res = df_subset.loc[0]
    assert type(res) == pd.Series
    if 'geometry' in column_set:
        assert not isinstance(res.geometry, pd.Series)
        assert res.geometry == Point(0, 0)