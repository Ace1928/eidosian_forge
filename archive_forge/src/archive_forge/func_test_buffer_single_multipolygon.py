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
def test_buffer_single_multipolygon():
    multi_poly = shapely.geometry.MultiPolygon([shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)])
    arr = from_shapely([multi_poly])
    result = arr.buffer(1)
    expected = [multi_poly.buffer(1)]
    equal_geometries(result, expected)
    result = arr.buffer(np.array([1]))
    equal_geometries(result, expected)