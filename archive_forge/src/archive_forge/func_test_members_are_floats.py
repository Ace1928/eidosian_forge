import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_members_are_floats(self):
    t = Affine(1, 2, 3, 4, 5, 6)
    for m in t:
        assert isinstance(m, float), repr(m)