import pandas as pd
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
from geopandas.tools import geocode, reverse_geocode
from geopandas.tools.geocoding import _prepare_geocode_result
from geopandas.tests.util import assert_geoseries_equal, mock
from pandas.testing import assert_series_equal
from geopandas.testing import assert_geodataframe_equal
import pytest
def test_bad_provider_forward():
    from geopy.exc import GeocoderNotFound
    with pytest.raises(GeocoderNotFound):
        geocode(['cambridge, ma'], 'badprovider')