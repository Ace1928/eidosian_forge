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
def test_covers(self):
    res = self.g7.covers(self.g8)
    exp = Series([True, False])
    assert_series_equal(res, exp)
    expected = [False, True, True, True, True, True, False, False]
    with pytest.warns(UserWarning, match='The indices .+ different'):
        assert_array_dtype_equal(expected, self.g0.covers(self.g9, align=True))
    expected = [False, False, True, False, False, False, False]
    assert_array_dtype_equal(expected, self.g0.covers(self.g9, align=False))