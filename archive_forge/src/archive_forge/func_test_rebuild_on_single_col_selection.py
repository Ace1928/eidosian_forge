from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_rebuild_on_single_col_selection(self):
    """Selecting a single column should not rebuild the spatial index."""
    original_index = self.df.sindex
    geometry_col = self.df['geom']
    assert geometry_col.sindex is original_index
    geometry_col = self.df.geometry
    assert geometry_col.sindex is original_index