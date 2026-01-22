import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_non_overlapping(how):
    p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    p2 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
    df1 = GeoDataFrame({'col1': [1], 'geometry': [p1]})
    df2 = GeoDataFrame({'col2': [2], 'geometry': [p2]})
    result = overlay(df1, df2, how=how)
    if how == 'intersection':
        if PANDAS_GE_20:
            index = None
        else:
            index = pd.Index([], dtype='object')
        expected = GeoDataFrame({'col1': np.array([], dtype='int64'), 'col2': np.array([], dtype='int64'), 'geometry': []}, index=index)
    elif how == 'union':
        expected = GeoDataFrame({'col1': [1, np.nan], 'col2': [np.nan, 2], 'geometry': [p1, p2]})
    elif how == 'identity':
        expected = GeoDataFrame({'col1': [1.0], 'col2': [np.nan], 'geometry': [p1]})
    elif how == 'symmetric_difference':
        expected = GeoDataFrame({'col1': [1, np.nan], 'col2': [np.nan, 2], 'geometry': [p1, p2]})
    elif how == 'difference':
        expected = GeoDataFrame({'col1': [1], 'geometry': [p1]})
    assert_geodataframe_equal(result, expected)