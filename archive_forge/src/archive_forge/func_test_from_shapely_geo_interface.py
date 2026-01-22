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
def test_from_shapely_geo_interface():

    class Point:

        def __init__(self, x, y):
            self.x = x
            self.y = y

        @property
        def __geo_interface__(self):
            return {'type': 'Point', 'coordinates': (self.x, self.y)}
    result = from_shapely([Point(1.0, 2.0), Point(3.0, 4.0)])
    expected = from_shapely([shapely.geometry.Point(1.0, 2.0), shapely.geometry.Point(3.0, 4.0)])
    assert all((v.equals(t) for v, t in zip(result, expected)))