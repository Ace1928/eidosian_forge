import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
def df_epsg26918():
    return _create_df(x=range(-1683723, -1683723 + 10, 1), y=range(6689139, 6689139 + 10, 1), crs='epsg:26918')