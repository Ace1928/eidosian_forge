import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_multiple_geoms(self):
    arr = from_shapely(self.geoms, crs=27700)
    s = GeoSeries(self.geoms, crs=4326)
    df = GeoDataFrame(s, geometry=arr, columns=['col1'])
    assert df.crs == self.osgb
    assert df.geometry.crs == self.osgb
    assert df.geometry.values.crs == self.osgb
    assert df.col1.crs == self.wgs
    assert df.col1.values.crs == self.wgs