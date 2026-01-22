import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_transform_precision():
    t = Affine.rotation(45.0)
    assert t.precision == EPSILON
    t.precision = 1e-10
    assert t.precision == 1e-10
    assert Affine.rotation(0.0).precision == EPSILON