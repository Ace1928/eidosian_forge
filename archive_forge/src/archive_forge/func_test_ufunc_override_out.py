import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
def test_ufunc_override_out(self):

    class A:

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            return kwargs

    class B:

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            return kwargs
    a = A()
    b = B()
    res0 = np.multiply(a, b, 'out_arg')
    res1 = np.multiply(a, b, out='out_arg')
    res2 = np.multiply(2, b, 'out_arg')
    res3 = np.multiply(3, b, out='out_arg')
    res4 = np.multiply(a, 4, 'out_arg')
    res5 = np.multiply(a, 5, out='out_arg')
    assert_equal(res0['out'][0], 'out_arg')
    assert_equal(res1['out'][0], 'out_arg')
    assert_equal(res2['out'][0], 'out_arg')
    assert_equal(res3['out'][0], 'out_arg')
    assert_equal(res4['out'][0], 'out_arg')
    assert_equal(res5['out'][0], 'out_arg')
    res6 = np.modf(a, 'out0', 'out1')
    res7 = np.frexp(a, 'out0', 'out1')
    assert_equal(res6['out'][0], 'out0')
    assert_equal(res6['out'][1], 'out1')
    assert_equal(res7['out'][0], 'out0')
    assert_equal(res7['out'][1], 'out1')
    assert_(np.sin(a, None) == {})
    assert_(np.sin(a, out=None) == {})
    assert_(np.sin(a, out=(None,)) == {})
    assert_(np.modf(a, None) == {})
    assert_(np.modf(a, None, None) == {})
    assert_(np.modf(a, out=(None, None)) == {})
    with assert_raises(TypeError):
        np.modf(a, out=None)
    assert_raises(TypeError, np.multiply, a, b, 'one', out='two')
    assert_raises(TypeError, np.multiply, a, b, 'one', 'two')
    assert_raises(ValueError, np.multiply, a, b, out=('one', 'two'))
    assert_raises(TypeError, np.multiply, a, out=())
    assert_raises(TypeError, np.modf, a, 'one', out=('two', 'three'))
    assert_raises(TypeError, np.modf, a, 'one', 'two', 'three')
    assert_raises(ValueError, np.modf, a, out=('one', 'two', 'three'))
    assert_raises(ValueError, np.modf, a, out=('one',))