import unittest
from math import pi
import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Point
from shapely.wkt import loads as load_wkt
def test_geom(g2, g3=None):
    assert not g2.has_z
    a2 = affinity.affine_transform(g2, matrix2d)
    assert not a2.has_z
    assert g2.equals(a2)
    if g3 is not None:
        assert g3.has_z
        a3 = affinity.affine_transform(g3, matrix3d)
        assert a3.has_z
        assert g3.equals(a3)
    return