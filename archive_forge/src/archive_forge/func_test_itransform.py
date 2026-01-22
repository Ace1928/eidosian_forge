import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_itransform(self):
    pts = [(4, 1), (-1, 0), (3, 2)]
    r = Affine.scale(-2).itransform(pts)
    assert r is None, r
    assert pts == [(-8, -2), (2, 0), (-6, -4)]
    A = Affine.rotation(33)
    pts = [(4, 1), (-1, 0), (3, 2)]
    pts_expect = [A * pt for pt in pts]
    r = A.itransform(pts)
    assert r is None
    assert pts == pts_expect