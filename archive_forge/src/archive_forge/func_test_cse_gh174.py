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
def test_cse_gh174():
    x = se.symbols('x')
    funcs = [se.cos(x) ** i for i in range(5)]
    f_lmb = se.Lambdify([x], funcs)
    f_cse = se.Lambdify([x], funcs, cse=True)
    a = np.array([1, 2, 3])
    assert np.allclose(f_lmb(a), f_cse(a))