import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_skip_exact_same():
    df = df_epsg26918()
    utm = df.to_crs(df.crs)
    assert_geodataframe_equal(df, utm, check_less_precise=True)