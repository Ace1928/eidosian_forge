import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_is_conformal(self):
    assert Affine.identity().is_conformal
    assert Affine.scale(2.5, 6.1).is_conformal
    assert Affine.translation(4, -1).is_conformal
    assert Affine.rotation(90).is_conformal
    assert Affine.rotation(-26).is_conformal
    assert not Affine.shear(4, -1).is_conformal