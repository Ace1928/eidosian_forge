from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_query_bulk_invalid_input_geometry(self):
    """
        Tests the `query_bulk` method with invalid input for the `geometry` parameter.
        """
    test_array = 'notanarray'
    with pytest.raises(TypeError):
        self.df.sindex.query(test_array)