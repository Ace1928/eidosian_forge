import numpy as np
from shapely.geometry import Point
from shapely.wkt import loads
import geopandas
import pytest
from pandas.testing import assert_series_equal
def test_hilbert_distance():
    geoms = geopandas.GeoSeries.from_wkt(['POINT (0 0)', 'POINT (1 1)', 'POINT (1 0)', 'POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))'])
    result = geoms.hilbert_distance(total_bounds=(0, 0, 1, 1), level=2)
    assert result.tolist() == [0, 10, 15, 2]
    result = geoms.hilbert_distance(total_bounds=(0, 0, 1, 1), level=3)
    assert result.tolist() == [0, 42, 63, 10]
    result = geoms.hilbert_distance(total_bounds=(0, 0, 1, 1), level=16)
    assert result.tolist() == [0, 2863311530, 4294967295, 715827882]