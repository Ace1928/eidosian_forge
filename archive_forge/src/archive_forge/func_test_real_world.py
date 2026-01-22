import math
import unittest
from textwrap import dedent
import pytest
import affine
from affine import Affine, EPSILON
def test_real_world(self):
    s = dedent('                 39.9317755024\n                 30.0907511581\n                 30.0907511576\n                -39.9317755019\n            2658137.2266720217\n            5990821.7039887439')
    a1 = affine.loadsw(s)
    assert a1.almost_equals(Affine(39.931775502364644, 30.090751157602412, 2658102.2154086917, 30.090751157602412, -39.931775502364644, 5990826.624500916))
    a1out = affine.dumpsw(a1)
    assert isinstance(a1out, str)
    a2 = affine.loadsw(a1out)
    assert a1.almost_equals(a2)