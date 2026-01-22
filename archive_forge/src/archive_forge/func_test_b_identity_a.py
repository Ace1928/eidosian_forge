import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, LineString, GeometryCollection, box
import geopandas
from geopandas import GeoDataFrame, GeoSeries, overlay, read_file
from geopandas._compat import PANDAS_GE_20
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
def test_b_identity_a(self):
    df_result = overlay(self.layer_b, self.layer_a, how='identity')
    assert_geodataframe_equal(df_result, self.b_identity_a)