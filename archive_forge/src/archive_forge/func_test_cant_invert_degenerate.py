import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_cant_invert_degenerate(self):
    t = Affine.scale(0)
    with pytest.raises(affine.TransformNotInvertibleError):
        ~t