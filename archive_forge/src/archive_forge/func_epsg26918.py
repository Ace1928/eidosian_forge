import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
@pytest.fixture(params=[26918, 'epsg:26918', pytest.param({'init': 'epsg:26918', 'no_defs': True}), '+proj=utm +zone=18 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ', {'proj': 'utm', 'zone': 18, 'datum': 'NAD83', 'units': 'm', 'no_defs': True}], ids=['epsg_number', 'epsg_string', 'epsg_dict', 'proj4_string', 'proj4_dict'])
def epsg26918(request):
    if isinstance(request.param, int):
        return {'epsg': request.param}
    return {'crs': request.param}