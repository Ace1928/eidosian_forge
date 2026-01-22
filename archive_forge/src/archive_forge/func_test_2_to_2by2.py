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
def test_2_to_2by2():
    L, check = _get_2_to_2by2_list()
    inp = [13, 17]
    A = L(inp)
    check(A, inp)