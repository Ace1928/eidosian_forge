import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_eccentricity_complex():
    assert (Affine.scale(2, 3) * Affine.rotation(77)).eccentricity == pytest.approx(math.sqrt(5) / 3)
    assert (Affine.rotation(77) * Affine.scale(2, 3)).eccentricity == pytest.approx(math.sqrt(5) / 3)
    assert (Affine.translation(32, -47) * Affine.rotation(77) * Affine.scale(2, 3)).eccentricity == pytest.approx(math.sqrt(5) / 3)