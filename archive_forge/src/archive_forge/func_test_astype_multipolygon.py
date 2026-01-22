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
def test_astype_multipolygon():
    multi_poly = shapely.geometry.MultiPolygon([shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(3, 3, 4, 4)])
    arr = from_shapely([multi_poly])
    result = arr.astype(str)
    assert isinstance(result[0], str)
    assert result[0] == multi_poly.wkt
    result = arr.astype(object)
    assert isinstance(result[0], shapely.geometry.base.BaseGeometry)
    result = arr.astype(np.dtype('U10'))
    assert result.dtype == np.dtype('U10')
    assert result[0] == multi_poly.wkt[:10]