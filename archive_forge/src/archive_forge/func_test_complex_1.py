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
def test_complex_1():
    x = se.Symbol('x')
    lmb = se.Lambdify([x], [1j + x], real=False)
    assert abs(lmb([11 + 13j])[0] - (11 + 14j)) < 1e-15