import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
@pytest.mark.filterwarnings('ignore:Accessing CRS')
def test_crs_with_no_geom_fails(self):
    with pytest.raises(ValueError, match='Assigning CRS to a GeoDataFrame without'):
        df = GeoDataFrame()
        df.crs = 4326