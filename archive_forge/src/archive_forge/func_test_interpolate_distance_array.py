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
def test_interpolate_distance_array(self):
    expected = GeoSeries([Point(0.0, 0.75), Point(1.0, 0.5)])
    self._test_binary_topological('interpolate', expected, self.g5, np.array([0.75, 1.5]))
    expected = GeoSeries([Point(0.5, 1.0), Point(0.0, 1.0)])
    self._test_binary_topological('interpolate', expected, self.g5, np.array([0.75, 1.5]), normalized=True)