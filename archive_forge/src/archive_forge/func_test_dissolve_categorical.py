import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
def test_dissolve_categorical():
    gdf = geopandas.GeoDataFrame({'cat': pd.Categorical(['a', 'a', 'b', 'b']), 'noncat': [1, 1, 1, 2], 'to_agg': [1, 2, 3, 4], 'geometry': geopandas.array.from_wkt(['POINT (0 0)', 'POINT (1 1)', 'POINT (2 2)', 'POINT (3 3)'])})
    expected_gdf_observed_false = geopandas.GeoDataFrame({'cat': pd.Categorical(['a', 'a', 'b', 'b']), 'noncat': [1, 2, 1, 2], 'geometry': geopandas.array.from_wkt(['MULTIPOINT (0 0, 1 1)', None, 'POINT (2 2)', 'POINT (3 3)']), 'to_agg': [1, None, 3, 4]}).set_index(['cat', 'noncat'])
    expected_gdf_observed_true = geopandas.GeoDataFrame({'cat': pd.Categorical(['a', 'b', 'b']), 'noncat': [1, 1, 2], 'geometry': geopandas.array.from_wkt(['MULTIPOINT (0 0, 1 1)', 'POINT (2 2)', 'POINT (3 3)']), 'to_agg': [1, 3, 4]}).set_index(['cat', 'noncat'])
    assert_frame_equal(expected_gdf_observed_false, gdf.dissolve(['cat', 'noncat']))
    assert_frame_equal(expected_gdf_observed_true, gdf.dissolve(['cat', 'noncat'], observed=True))