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
def test_explode_geometry_name_two_geoms(self):
    s = GeoSeries([MultiPoint([Point(1, 2), Point(2, 3)]), Point(5, 5)])
    df = GeoDataFrame({'col': [1, 2], 'geom': s, 'geometry': s}, geometry='geom')
    test_df = df.explode(index_parts=True)
    assert test_df.geometry.name == 'geom'
    assert test_df.geometry.name == test_df._geometry_column_name
    assert 'geometry' in test_df.columns