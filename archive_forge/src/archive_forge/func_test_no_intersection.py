import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_no_intersection():
    gs = GeoSeries([Point(x, x).buffer(0.1) for x in range(3)])
    gdf1 = GeoDataFrame({'foo': ['a', 'b', 'c']}, geometry=gs)
    gdf2 = GeoDataFrame({'bar': ['1', '3', '5']}, geometry=gs.translate(1))
    expected = GeoDataFrame(columns=['foo', 'bar', 'geometry'])
    result = overlay(gdf1, gdf2, how='intersection')
    assert_geodataframe_equal(result, expected, check_index_type=False)