import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
@pytest.fixture(params=[4326, 'epsg:4326', pytest.param({'init': 'epsg:4326'}), '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs', {'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs': True}], ids=['epsg_number', 'epsg_string', 'epsg_dict', 'proj4_string', 'proj4_dict'])
def epsg4326(request):
    if isinstance(request.param, int):
        return {'epsg': request.param}
    return {'crs': request.param}