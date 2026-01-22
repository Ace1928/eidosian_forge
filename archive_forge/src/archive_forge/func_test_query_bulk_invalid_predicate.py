from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_query_bulk_invalid_predicate(self):
    """Tests the `query_bulk` method with invalid predicates."""
    test_geom_bounds = (-1, -1, -0.5, -0.5)
    test_predicate = 'test'
    test_geom = geopandas.GeoSeries([box(*test_geom_bounds)], index=['0'])
    with pytest.raises(ValueError):
        self.df.sindex.query(test_geom.geometry, predicate=test_predicate)