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
def test_Lambdify_Ndimensional_order_F():
    args, nd_exprs_a, nd_exprs_b, f_a, f_b = _get_Ndim_args_exprs_funcs(order='F')
    lmb4 = se.Lambdify(args, nd_exprs_a, nd_exprs_b, order='F')
    nargs = len(args)
    inp_extra_shape = (3, 5, 4)
    inp_shape = (nargs,) + inp_extra_shape
    inp4 = np.arange(reduce(mul, inp_shape) * 1.0).reshape(inp_shape, order='F')
    out4a, out4b = lmb4(inp4)
    assert out4a.ndim == 7
    assert out4a.shape == nd_exprs_a.shape + inp_extra_shape
    assert out4b.ndim == 6
    assert out4b.shape == nd_exprs_b.shape + inp_extra_shape
    raises(ValueError, lambda: lmb4(inp4.T))
    for b, c, d in np.ndindex(inp_extra_shape):
        _x, _y = inp4[:, b, c, d]
        for index in np.ndindex(*nd_exprs_a.shape):
            assert np.isclose(out4a[index + (b, c, d)], f_a(index, _x, _y))
        for index in np.ndindex(*nd_exprs_b.shape):
            assert np.isclose(out4b[index + (b, c, d)], f_b(index, _x, _y))