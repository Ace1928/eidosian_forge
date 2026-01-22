import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
def test_affine_params(self):
    g = load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
    with pytest.raises(TypeError):
        affinity.affine_transform(g, None)
    with pytest.raises(ValueError):
        affinity.affine_transform(g, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    with pytest.raises(AttributeError):
        affinity.affine_transform(None, [1, 2, 3, 4, 5, 6])