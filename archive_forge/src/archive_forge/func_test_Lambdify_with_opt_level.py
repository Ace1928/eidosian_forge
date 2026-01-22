import array
import cmath
from functools import reduce
import itertools
from operator import mul
import math
import symengine as se
from symengine.test_utilities import raises
from symengine import have_numpy
import unittest
from unittest.case import SkipTest
@unittest.skipUnless(have_numpy, 'Numpy not installed')
def test_Lambdify_with_opt_level():
    args = x, y, z = se.symbols('x y z')
    raises(TypeError, lambda: se.Lambdify(args, [x + y + z, x ** 2, (x - y) / z, x * y * z], backend='lambda', opt_level=0))