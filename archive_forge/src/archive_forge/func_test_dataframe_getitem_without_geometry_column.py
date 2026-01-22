import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def test_dataframe_getitem_without_geometry_column(self):
    df = GeoDataFrame({'col': range(10)}, geometry=self.arr)
    df['geom2'] = df.geometry.centroid
    subset = df[['col', 'geom2']]
    with pytest.raises(AttributeError, match='The CRS attribute of a GeoDataFrame without an active'):
        assert subset.crs == self.osgb