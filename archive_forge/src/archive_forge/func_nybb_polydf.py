import warnings
import numpy as np
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file
from pandas.testing import assert_frame_equal
import pytest
from geopandas._compat import PANDAS_GE_15, PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, geom_almost_equals
@pytest.fixture
def nybb_polydf():
    nybb_filename = geopandas.datasets.get_path('nybb')
    nybb_polydf = read_file(nybb_filename)
    nybb_polydf = nybb_polydf[['geometry', 'BoroName', 'BoroCode']]
    nybb_polydf = nybb_polydf.rename(columns={'geometry': 'myshapes'})
    nybb_polydf = nybb_polydf.set_geometry('myshapes')
    nybb_polydf['manhattan_bronx'] = 5
    nybb_polydf.loc[3:4, 'manhattan_bronx'] = 6
    nybb_polydf['BoroCode'] = nybb_polydf['BoroCode'].astype('int64')
    return nybb_polydf