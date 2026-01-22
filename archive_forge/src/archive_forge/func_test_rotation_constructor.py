import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_rotation_constructor(self):
    rot = Affine.rotation(60)
    assert isinstance(rot, Affine)
    r = math.radians(60)
    s, c = (math.sin(r), math.cos(r))
    assert tuple(rot) == (c, -s, 0, s, c, 0, 0, 0, 1)
    rot = Affine.rotation(337)
    r = math.radians(337)
    s, c = (math.sin(r), math.cos(r))
    seq_almost_equal(tuple(rot), (c, -s, 0, s, c, 0, 0, 0, 1))
    assert tuple(Affine.rotation(0)) == tuple(Affine.identity())