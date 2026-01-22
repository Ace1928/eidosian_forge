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
@pytest.mark.skipif(compat.USE_PYGEOS or compat.USE_SHAPELY_20, reason='segmentize keyword introduced in shapely 2.0')
def test_segmentize_shapely_pre20(self):
    s = GeoSeries([Point(1, 1)])
    with pytest.raises(NotImplementedError, match=f'shapely >= 2.0 or PyGEOS is required, version {shapely.__version__} is installed'):
        s.segmentize(max_segment_length=1)