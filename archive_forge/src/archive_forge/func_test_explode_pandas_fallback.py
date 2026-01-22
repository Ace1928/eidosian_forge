import string
import warnings
import numpy as np
from numpy.testing import assert_array_equal
from pandas import DataFrame, Index, MultiIndex, Series, concat
import shapely
from shapely.geometry import (
from shapely.geometry.collection import GeometryCollection
from shapely.ops import unary_union
from shapely import wkt
from geopandas import GeoDataFrame, GeoSeries
from geopandas.base import GeoPandasBase
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
from geopandas.tests.util import assert_geoseries_equal, geom_equals
from geopandas import _compat as compat
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
import pytest
def test_explode_pandas_fallback(self):
    d = {'col1': [['name1', 'name2'], ['name3', 'name4']], 'geometry': [MultiPoint([(1, 2), (3, 4)]), MultiPoint([(2, 1), (0, 0)])]}
    gdf = GeoDataFrame(d, crs=4326)
    expected_df = GeoDataFrame({'col1': ['name1', 'name2', 'name3', 'name4'], 'geometry': [MultiPoint([(1, 2), (3, 4)]), MultiPoint([(1, 2), (3, 4)]), MultiPoint([(2, 1), (0, 0)]), MultiPoint([(2, 1), (0, 0)])]}, index=[0, 0, 1, 1], crs=4326)
    exploded_df = gdf.explode('col1')
    assert_geodataframe_equal(exploded_df, expected_df)
    exploded_df = gdf.explode(column='col1')
    assert_geodataframe_equal(exploded_df, expected_df)