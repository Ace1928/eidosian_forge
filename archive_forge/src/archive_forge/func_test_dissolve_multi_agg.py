import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
def test_dissolve_multi_agg(nybb_polydf, merged_shapes):
    merged_shapes['BoroCode', 'min'] = [3, 1]
    merged_shapes['BoroCode', 'max'] = [5, 2]
    merged_shapes['BoroName', 'count'] = [3, 2]
    with warnings.catch_warnings(record=True) as record:
        test = nybb_polydf.dissolve(by='manhattan_bronx', aggfunc={'BoroCode': ['min', 'max'], 'BoroName': 'count'})
    assert_geodataframe_equal(test, merged_shapes)
    assert len(record) == 0