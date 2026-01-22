import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_rotation_constructor_with_pivot(self):
    assert tuple(Affine.rotation(60)) == tuple(Affine.rotation(60, pivot=(0, 0)))
    rot = Affine.rotation(27, pivot=(2, -4))
    r = math.radians(27)
    s, c = (math.sin(r), math.cos(r))
    assert tuple(rot) == (c, -s, 2 - 2 * c - 4 * s, s, c, -4 - 2 * s + 4 * c, 0, 0, 1)
    assert tuple(Affine.rotation(0, (-3, 2))) == tuple(Affine.identity())