import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
def test_dissolve_sort():
    gdf = geopandas.GeoDataFrame({'a': [2, 1, 1], 'geometry': geopandas.array.from_wkt(['POINT (0 0)', 'POINT (1 1)', 'POINT (2 2)'])})
    expected_unsorted = geopandas.GeoDataFrame({'a': [2, 1], 'geometry': geopandas.array.from_wkt(['POINT (0 0)', 'MULTIPOINT (1 1, 2 2)'])}).set_index('a')
    expected_sorted = expected_unsorted.sort_index()
    assert_frame_equal(expected_sorted, gdf.dissolve('a'))
    assert_frame_equal(expected_unsorted, gdf.dissolve('a', sort=False))