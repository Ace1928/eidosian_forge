import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_empty_overlay_return_non_duplicated_columns():
    nybb = geopandas.read_file(geopandas.datasets.get_path('nybb'))
    nybb2 = nybb.copy()
    nybb2.geometry = nybb2.translate(20000000)
    result = geopandas.overlay(nybb, nybb2)
    expected = GeoDataFrame(columns=['BoroCode_1', 'BoroName_1', 'Shape_Leng_1', 'Shape_Area_1', 'BoroCode_2', 'BoroName_2', 'Shape_Leng_2', 'Shape_Area_2', 'geometry'], crs=nybb.crs)
    assert_geodataframe_equal(result, expected, check_dtype=False)