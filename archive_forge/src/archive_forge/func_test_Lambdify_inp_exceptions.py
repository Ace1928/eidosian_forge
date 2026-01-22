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
def test_Lambdify_inp_exceptions():
    args = x, y = se.symbols('x y')
    lmb1 = se.Lambdify([x], x ** 2)
    raises(ValueError, lambda: lmb1([]))
    assert lmb1(4) == 16
    assert np.all(lmb1([4, 2]) == [16, 4])
    lmb2 = se.Lambdify(args, x ** 2 + y ** 2)
    assert lmb2([2, 3]) == 13
    raises(ValueError, lambda: lmb2([]))
    raises(ValueError, lambda: lmb2([2]))
    raises(ValueError, lambda: lmb2([2, 3, 4]))
    assert np.all(lmb2([2, 3, 4, 5]) == [13, 16 + 25])

    def _mtx(_x, _y):
        return [[_x - _y, _y ** 2], [_x + _y, _x ** 2], [_x * _y, _x ** _y]]
    mtx = np.array(_mtx(x, y), order='F')
    lmb3 = se.Lambdify(args, mtx, order='F')
    inp3a = [2, 3]
    assert np.all(lmb3(inp3a) == _mtx(*inp3a))
    inp3b = np.array([2, 3, 4, 5, 3, 2, 1, 5])
    for inp in [inp3b, inp3b.tolist(), inp3b.reshape((2, 4), order='F')]:
        out3b = lmb3(inp)
        assert out3b.shape == (3, 2, 4)
        for i in range(4):
            assert np.all(out3b[..., i] == _mtx(*inp3b[2 * i:2 * (i + 1)]))
    raises(ValueError, lambda: lmb3(inp3b.reshape((4, 2))))
    raises(ValueError, lambda: lmb3(inp3b.reshape((2, 4)).T))