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
def test_Lambdify_Ndimensional_order_C():
    args, nd_exprs_a, nd_exprs_b, f_a, f_b = _get_Ndim_args_exprs_funcs(order='C')
    lmb4 = se.Lambdify(args, nd_exprs_a, nd_exprs_b, order='C')
    nargs = len(args)
    inp_extra_shape = (3, 5, 4)
    inp_shape = inp_extra_shape + (nargs,)
    inp4 = np.arange(reduce(mul, inp_shape) * 1.0).reshape(inp_shape, order='C')
    out4a, out4b = lmb4(inp4)
    assert out4a.ndim == 7
    assert out4a.shape == inp_extra_shape + nd_exprs_a.shape
    assert out4b.ndim == 6
    assert out4b.shape == inp_extra_shape + nd_exprs_b.shape
    raises(ValueError, lambda: lmb4(inp4.T))
    for b, c, d in np.ndindex(inp_extra_shape):
        _x, _y = inp4[b, c, d, :]
        for index in np.ndindex(*nd_exprs_a.shape):
            assert np.isclose(out4a[(b, c, d) + index], f_a(index, _x, _y))
        for index in np.ndindex(*nd_exprs_b.shape):
            assert np.isclose(out4b[(b, c, d) + index], f_b(index, _x, _y))