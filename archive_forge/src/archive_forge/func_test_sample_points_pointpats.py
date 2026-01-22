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
@pytest.mark.skipif(not (compat.USE_PYGEOS or compat.USE_SHAPELY_20), reason='get_coordinates not implemented for shapely<2')
@pytest.mark.parametrize('size', [10, 20, 50])
def test_sample_points_pointpats(self, size):
    pytest.importorskip('pointpats')
    for gs in (self.g1, self.na, self.a1):
        output = gs.sample_points(size, method='cluster_poisson')
        assert_index_equal(gs.index, output.index)
        assert len(output.explode(ignore_index=True)) == len(gs[~gs.is_empty]) * size
    with pytest.raises(AttributeError, match='pointpats.random module has no'):
        gs.sample_points(10, method='nonexistent')