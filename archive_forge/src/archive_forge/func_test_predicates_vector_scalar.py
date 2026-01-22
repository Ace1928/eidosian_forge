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
@pytest.mark.parametrize('attr,args', [('contains', ()), ('covers', ()), ('crosses', ()), ('disjoint', ()), ('geom_equals', ()), ('intersects', ()), ('overlaps', ()), ('touches', ()), ('within', ()), ('geom_equals_exact', (0.1,)), ('geom_almost_equals', (3,))])
def test_predicates_vector_scalar(attr, args):
    na_value = False
    point = points[0]
    tri = triangles[0]
    for other in [point, tri, shapely.geometry.Polygon()]:
        result = getattr(T, attr)(other, *args)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        expected = [getattr(tri, attr if 'geom' not in attr else attr[5:])(other, *args) if tri is not None else na_value for tri in triangles]
        assert result.tolist() == expected