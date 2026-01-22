import random
import numpy as np
import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point, Polygon, LineString
from geopandas import GeoSeries, GeoDataFrame, points_from_xy, datasets, read_file
from geopandas.array import from_shapely, from_wkb, from_wkt, GeometryArray
from geopandas.testing import assert_geodataframe_equal
@pytest.mark.parametrize('attr, arg', [('affine_transform', ([0, 1, 1, 0, 0, 0],)), ('translate', ()), ('rotate', (10,)), ('scale', ()), ('skew', ())])
def test_affinity_methods(self, attr, arg):
    result = getattr(self.arr, attr)(*arg)
    assert result.crs == self.osgb