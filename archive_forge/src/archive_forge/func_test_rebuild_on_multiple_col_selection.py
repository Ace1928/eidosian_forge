from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
def test_rebuild_on_multiple_col_selection(self):
    """Selecting a subset of columns preserves the index."""
    original_index = self.df.sindex
    subset1 = self.df[['geom', 'A']]
    if compat.PANDAS_GE_20 and (not pd.options.mode.copy_on_write):
        assert subset1.sindex is not original_index
    else:
        assert subset1.sindex is original_index
    subset2 = self.df[['A', 'geom']]
    if compat.PANDAS_GE_20 and (not pd.options.mode.copy_on_write):
        assert subset2.sindex is not original_index
    else:
        assert subset2.sindex is original_index