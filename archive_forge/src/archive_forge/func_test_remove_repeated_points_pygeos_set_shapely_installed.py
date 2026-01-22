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
@pytest.mark.skipif(shapely.geos.geos_version < (3, 11, 0), reason='requires GEOS>=3.11')
@pytest.mark.skipif(not (compat.USE_PYGEOS and compat.SHAPELY_GE_20), reason='remove_repeated_points is only implemented for shapely >= 2.0')
def test_remove_repeated_points_pygeos_set_shapely_installed(self):
    with pytest.warns(UserWarning, match='PyGEOS does not support remove_repeated_points, and Shapely >= 2 is installed'):
        self.g1.remove_repeated_points()