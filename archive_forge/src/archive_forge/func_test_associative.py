import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_associative():
    point = (12, 5)
    trans = Affine.translation(-10.0, -5.0)
    rot90 = Affine.rotation(90.0)
    result1 = rot90 * (trans * point)
    result2 = rot90 * trans * point
    seq_almost_equal(result1, (0.0, 2.0))
    seq_almost_equal(result1, result2)