import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_mul_by_identity(self):
    t = Affine(1, 2, 3, 4, 5, 6)
    assert tuple(t * Affine.identity()) == tuple(t)