import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas
@pytest.mark.parametrize('geom_name', ['geometry', pytest.param('geom', marks=pytest.mark.xfail(reason='pre-regression behaviour only works for geometry col geometry'))])
def test_loc_add_row(geom_name):
    nybb_filename = geopandas.datasets.get_path('nybb')
    nybb = geopandas.read_file(nybb_filename)[['BoroCode', 'geometry']]
    if geom_name != 'geometry':
        nybb = nybb.rename_geometry(geom_name)
    nybb.loc[5] = [6, nybb.geometry.iloc[0]]
    assert nybb.geometry.dtype == 'geometry'
    assert nybb.crs is None