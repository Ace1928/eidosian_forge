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
def test_sort_values_empty_missing():
    s = GeoSeries([Point(0, 0), None, Point(), Point(1, 1)])
    res = s.sort_values()
    assert res.index.tolist() == [2, 0, 3, 1]
    res = s.sort_values(ascending=False)
    assert res.index.tolist() == [3, 0, 2, 1]
    res = s.sort_values(na_position='first')
    assert res.index.tolist() == [1, 2, 0, 3]
    res = s.sort_values(ascending=False, na_position='first')
    assert res.index.tolist() == [1, 3, 0, 2]
    s = GeoSeries([None, None, None])
    res = s.sort_values()
    assert res.index.tolist() == [0, 1, 2]
    s = GeoSeries([Point(), Point(), Point()])
    res = s.sort_values()
    assert res.index.tolist() == [0, 1, 2]
    s = GeoSeries([Point(), None, Point()])
    res = s.sort_values()
    assert res.index.tolist() == [0, 2, 1]