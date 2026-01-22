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
def test_binary_relate():
    attr = 'relate'
    na_value = None
    result = getattr(P[:len(T)], attr)(T[::-1])
    expected = [getattr(p, attr)(t) if t is not None and p is not None else na_value for t, p in zip(triangles[::-1], points)]
    assert list(result) == expected
    p = points[0]
    result = getattr(T, attr)(p)
    expected = [getattr(t, attr)(p) if t is not None else na_value for t in triangles]
    assert list(result) == expected