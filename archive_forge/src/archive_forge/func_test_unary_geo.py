import random
import numpy as np
import pandas as pd
from pyproj import CRS
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE, BaseGeometry
import shapely.wkb
import shapely.wkt
import geopandas
from geopandas.array import (
import geopandas._compat as compat
import pytest
@pytest.mark.parametrize('attr', ['boundary', 'centroid', 'convex_hull', 'envelope', 'exterior'])
def test_unary_geo(attr):
    na_value = None
    result = getattr(T, attr)
    expected = [getattr(t, attr) if t is not None else na_value for t in triangles]
    assert equal_geometries(result, expected)