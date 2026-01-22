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
def test_buffer_args(self):
    args = {'cap_style': 3, 'join_style': 2, 'mitre_limit': 2.5}
    calculated_series = self.g0.buffer(10, **args)
    for original, calculated in zip(self.g0, calculated_series):
        if original is None:
            assert calculated is None
        else:
            expected = original.buffer(10, **args)
            assert calculated.equals(expected)