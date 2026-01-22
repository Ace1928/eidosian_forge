import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_scale_constructor(self):
    scale = Affine.scale(5)
    assert isinstance(scale, Affine)
    assert tuple(scale) == (5, 0, 0, 0, 5, 0, 0, 0, 1)
    scale = Affine.scale(-1, 2)
    assert tuple(scale) == (-1, 0, 0, 0, 2, 0, 0, 0, 1)
    assert tuple(Affine.scale(1)) == tuple(Affine.identity())