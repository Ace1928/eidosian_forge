import pandas as pd
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
from geopandas.tools import geocode, reverse_geocode
from geopandas.tools.geocoding import _prepare_geocode_result
from geopandas.tests.util import assert_geoseries_equal, mock
from pandas.testing import assert_series_equal
from geopandas.testing import assert_geodataframe_equal
import pytest
def test_prepare_result_none():
    p0 = Point(12.3, -45.6)
    d = {'a': ('address0', p0.coords[0]), 'b': (None, None)}
    df = _prepare_geocode_result(d)
    assert type(df) is GeoDataFrame
    assert df.crs == 'EPSG:4326'
    assert len(df) == 2
    assert 'address' in df
    row = df.loc['b']
    assert row['geometry'].is_empty
    assert row['address'] is None