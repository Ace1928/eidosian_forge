import math
from typing import Sequence
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, GeometryCollection
import geopandas
import geopandas._compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, sjoin, sjoin_nearest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest
def test_sjoin_inner(self):
    countries = self.world[['geometry', 'name']]
    countries = countries.rename(columns={'name': 'country'})
    cities_with_country = sjoin(self.cities, countries, how='inner', predicate='intersects')
    assert cities_with_country.shape == (213, 4)