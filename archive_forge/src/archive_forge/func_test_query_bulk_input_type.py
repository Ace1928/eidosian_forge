from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.parametrize('predicate, test_geom, expected', ((None, (-1, -1, -0.5, -0.5), [[], []]), ('intersects', (-1, -1, -0.5, -0.5), [[], []]), ('contains', (-1, -1, 1, 1), [[0], [0]])))
def test_query_bulk_input_type(self, predicate, test_geom, expected):
    """Tests that query_bulk can accept a GeoSeries, GeometryArray or
        numpy array.
        """
    test_geom = geopandas.GeoSeries([box(*test_geom)], index=['0'])
    res = self.df.sindex.query(test_geom, predicate=predicate)
    assert_array_equal(res, expected)
    res = self.df.sindex.query(test_geom.geometry, predicate=predicate)
    assert_array_equal(res, expected)
    res = self.df.sindex.query(test_geom.geometry.values, predicate=predicate)
    assert_array_equal(res, expected)
    res = self.df.sindex.query(test_geom.geometry.values.to_numpy(), predicate=predicate)
    assert_array_equal(res, expected)
    res = self.df.sindex.query(test_geom.geometry.values.to_numpy(), predicate=predicate)
    assert_array_equal(res, expected)