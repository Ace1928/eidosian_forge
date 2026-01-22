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
@unittest.skipUnless(have_sympy, 'SymPy not installed')
def test_lambdify__sympy():
    import sympy as sp
    _sympy_lambdify_heterogeneous_output(se.lambdify, se.DenseMatrix)
    _sympy_lambdify_heterogeneous_output(sp.lambdify, sp.Matrix)