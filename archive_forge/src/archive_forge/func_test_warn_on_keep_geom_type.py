import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_warn_on_keep_geom_type(dfs):
    df1, df2 = dfs
    polys3 = GeoSeries([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), Polygon([(-1, 1), (1, 1), (1, 3), (-1, 3)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])])
    df3 = GeoDataFrame({'geometry': polys3})
    with pytest.warns(UserWarning, match='`keep_geom_type=True` in overlay'):
        overlay(df2, df3, keep_geom_type=None)