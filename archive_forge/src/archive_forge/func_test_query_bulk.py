from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.parametrize('predicate, test_geom, expected', ((None, [(-1, -1, -0.5, -0.5)], [[], []]), (None, [(-0.5, -0.5, 0.5, 0.5)], [[0], [0]]), (None, [(0, 0, 1, 1)], [[0, 0], [0, 1]]), ('intersects', [(-1, -1, -0.5, -0.5)], [[], []]), ('intersects', [(-0.5, -0.5, 0.5, 0.5)], [[0], [0]]), ('intersects', [(0, 0, 1, 1)], [[0, 0], [0, 1]]), ('intersects', [(-1, -1, -0.5, -0.5), (-0.5, -0.5, 0.5, 0.5)], [[1], [0]]), ('intersects', [(-1, -1, 1, 1), (-0.5, -0.5, 0.5, 0.5)], [[0, 0, 1], [0, 1, 0]]), ('within', [(0.25, 0.28, 0.75, 0.75)], [[], []]), ('within', [(0, 0, 10, 10)], [[], []]), ('within', [(11, 11, 12, 12)], [[0], [5]]), ('contains', [(0, 0, 1, 1)], [[], []]), ('contains', [(0, 0, 1.001, 1.001)], [[0], [1]]), ('contains', [(0.5, 0.5, 1.001, 1.001)], [[0], [1]]), ('contains', [(0.5, 0.5, 1.5, 1.5)], [[0], [1]]), ('contains', [(-1, -1, 2, 2)], [[0, 0], [0, 1]]), ('contains', [(10, 10, 20, 20)], [[0], [5]]), ('touches', [(-1, -1, 0, 0)], [[0], [0]]), ('touches', [(-0.5, -0.5, 1.5, 1.5)], [[], []]), ('covers', [(-0.5, -0.5, 1, 1)], [[0, 0], [0, 1]]), ('covers', [(0.001, 0.001, 0.99, 0.99)], [[], []]), ('covers', [(0, 0, 1, 1)], [[0, 0], [0, 1]]), ('contains_properly', [(0, 0, 1, 1)], [[], []]), ('contains_properly', [(0, 0, 1.001, 1.001)], [[0], [1]]), ('contains_properly', [(0.5, 0.5, 1.001, 1.001)], [[0], [1]]), ('contains_properly', [(0.5, 0.5, 1.5, 1.5)], [[0], [1]]), ('contains_properly', [(-1, -1, 2, 2)], [[0, 0], [0, 1]]), ('contains_properly', [(10, 10, 20, 20)], [[], []])))
def test_query_bulk(self, predicate, test_geom, expected):
    """Tests the `query_bulk` method with valid
        inputs and valid predicates.
        """
    test_geom = geopandas.GeoSeries([box(*geom) for geom in test_geom], index=range(len(test_geom)))
    res = self.df.sindex.query(test_geom, predicate=predicate)
    assert_array_equal(res, expected)