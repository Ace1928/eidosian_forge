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
def test_predicates_vector_vector(attr, args):
    na_value = False
    empty_value = True if attr == 'disjoint' else False
    A = [shapely.geometry.Polygon(), None] + [shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)]) for _ in range(100)] + [None]
    B = [shapely.geometry.Polygon([(random.random(), random.random()) for i in range(3)]) for _ in range(100)] + [shapely.geometry.Polygon(), None, None]
    vec_A = from_shapely(A)
    vec_B = from_shapely(B)
    result = getattr(vec_A, attr)(vec_B, *args)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool
    expected = []
    for a, b in zip(A, B):
        if a is None or b is None:
            expected.append(na_value)
        elif a.is_empty or b.is_empty:
            expected.append(empty_value)
        else:
            expected.append(getattr(a, attr if 'geom' not in attr else attr[5:])(b, *args))
    assert result.tolist() == expected