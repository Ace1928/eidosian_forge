import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_dataframe_setitem(self):
    arr = from_shapely(self.geoms)
    s = GeoSeries(arr, crs=27700)
    df = GeoDataFrame()
    with pytest.warns(FutureWarning, match="You are adding a column named 'geometry'"):
        df['geometry'] = s
    assert df.crs == self.osgb
    assert df.geometry.crs == self.osgb
    assert df.geometry.values.crs == self.osgb
    arr = from_shapely(self.geoms, crs=27700)
    df = GeoDataFrame()
    with pytest.warns(FutureWarning, match="You are adding a column named 'geometry'"):
        df['geometry'] = arr
    assert df.crs == self.osgb
    assert df.geometry.crs == self.osgb
    assert df.geometry.values.crs == self.osgb
    arr = from_shapely(self.geoms)
    df = GeoDataFrame({'col1': [1, 2], 'geometry': arr}, crs=4326)
    df['geometry'] = df['geometry'].to_crs(27700)
    assert df.crs == self.osgb
    assert df.geometry.crs == self.osgb
    assert df.geometry.values.crs == self.osgb
    arr = from_shapely(self.geoms)
    df = GeoDataFrame({'col1': [1, 2], 'geometry': arr, 'other_geom': arr}, crs=4326)
    df['other_geom'] = from_shapely(self.geoms, crs=27700)
    assert df.crs == self.wgs
    assert df.geometry.crs == self.wgs
    assert df['geometry'].crs == self.wgs
    assert df['other_geom'].crs == self.osgb