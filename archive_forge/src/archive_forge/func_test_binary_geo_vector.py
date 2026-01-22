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
@pytest.mark.parametrize('attr', ['difference', 'symmetric_difference', 'union', 'intersection'])
def test_binary_geo_vector(attr):
    na_value = None
    quads = [shapely.geometry.Polygon(), None]
    while len(quads) < 12:
        geom = shapely.geometry.Polygon([(random.random(), random.random()) for i in range(4)])
        if geom.is_valid:
            quads.append(geom)
    Q = from_shapely(quads)
    result = getattr(T, attr)(Q)
    expected = [getattr(t, attr)(q) if t is not None and q is not None else na_value for t, q in zip(triangles, quads)]
    assert equal_geometries(result, expected)