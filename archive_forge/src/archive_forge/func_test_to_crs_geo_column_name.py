import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_to_crs_geo_column_name():
    df = df_epsg26918()
    df = df.rename(columns={'geometry': 'geom'})
    df.set_geometry('geom', inplace=True)
    lonlat = df.to_crs(epsg=4326)
    utm = lonlat.to_crs(epsg=26918)
    assert lonlat.geometry.name == 'geom'
    assert utm.geometry.name == 'geom'
    assert_geodataframe_equal(df, utm, check_less_precise=True)