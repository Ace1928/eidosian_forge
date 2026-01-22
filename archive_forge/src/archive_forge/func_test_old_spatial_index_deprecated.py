from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skipif(not compat.HAS_RTREE, reason='no rtree installed')
def test_old_spatial_index_deprecated():
    t1 = Polygon([(0, 0), (1, 0), (1, 1)])
    t2 = Polygon([(0, 0), (1, 1), (0, 1)])
    stream = ((i, item.bounds, None) for i, item in enumerate([t1, t2]))
    with pytest.warns(FutureWarning):
        idx = geopandas.sindex.SpatialIndex(stream)
    assert list(idx.intersection((0, 0, 1, 1))) == [0, 1]