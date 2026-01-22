import os
from packaging.version import Version
import warnings
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import shapely
from shapely.geometry import Point, GeometryCollection, LineString, LinearRing
import geopandas
from geopandas import GeoDataFrame, GeoSeries
import geopandas._compat as compat
from geopandas.array import from_shapely
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
def test_df_apply_returning_series(df):
    result = df.apply(lambda row: row.geometry, axis=1)
    assert_geoseries_equal(result, df.geometry, check_crs=False)
    result = df.apply(lambda row: row.value1, axis=1)
    assert_series_equal(result, df['value1'].rename(None))
    result = df.apply(lambda x: float('NaN'), axis=1)
    assert result.dtype == 'float64'
    result = df.apply(lambda x: None, axis=1)
    assert result.dtype == 'object'
    res = df.apply(lambda row: df.geometry.to_frame(), axis=1)
    assert res.dtype == 'object'