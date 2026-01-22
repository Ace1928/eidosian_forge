import random
import unittest
from functools import partial
from itertools import islice
import pytest
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import MultiPolygon, Point
from shapely.ops import cascaded_union, unary_union
def test_cascaded_union(self):
    r = partial(random.uniform, -20.0, 20.0)
    points = [Point(r(), r()) for i in range(100)]
    spots = [p.buffer(2.5) for p in points]
    with pytest.warns(ShapelyDeprecationWarning, match='is deprecated'):
        u = cascaded_union(spots)
    assert u.geom_type in ('Polygon', 'MultiPolygon')