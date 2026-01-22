import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_to_crs_transform__missing_data():
    df = df_epsg26918()
    df.loc[3, 'geometry'] = None
    lonlat = df.to_crs(epsg=4326)
    utm = lonlat.to_crs(epsg=26918)
    assert_geodataframe_equal(df, utm, check_less_precise=True)