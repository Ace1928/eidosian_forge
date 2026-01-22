import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_mixed_geom_error():
    polys1 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    df1 = GeoDataFrame({'col1': [1, 2], 'geometry': polys1})
    mixed = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), LineString([(3, 3), (5, 3), (5, 5), (3, 5)])])
    dfmixed = GeoDataFrame({'col1': [1, 2], 'geometry': mixed})
    with pytest.raises(NotImplementedError):
        overlay(df1, dfmixed, keep_geom_type=True)