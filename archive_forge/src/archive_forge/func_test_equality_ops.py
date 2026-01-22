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
def test_equality_ops():
    with pytest.raises(ValueError):
        P[:5] == P[:7]
    a1 = from_shapely([points[1], points[2], points[3]])
    a2 = from_shapely([points[1], points[0], points[3]])
    res = a1 == a2
    assert res.tolist() == [True, False, True]
    res = a1 != a2
    assert res.tolist() == [False, True, False]
    multi_poly = shapely.geometry.MultiPolygon([shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)])
    a3 = from_shapely([points[1], points[2], points[3], multi_poly])
    res = a3 == multi_poly
    assert res.tolist() == [False, False, False, True]