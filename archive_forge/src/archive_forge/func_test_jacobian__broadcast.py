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
def test_jacobian__broadcast():
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x ** 3 * y, (x + 1) * (y + 1)])
    jac = v.jacobian(args)
    lmb = se.Lambdify(args, jac)
    out = np.empty((3, 2, 2))
    inp0 = (7, 11)
    inp1 = (8, 13)
    inp2 = (5, 9)
    inp = np.array([inp0, inp1, inp2])
    lmb(inp, out=out)
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        assert np.allclose(out[idx, ...], [[3 * X ** 2 * Y, X ** 3], [Y + 1, X + 1]])