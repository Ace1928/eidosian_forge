import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_reduce_arguments(self):
    f = np.add.reduce
    d = np.ones((5, 2), dtype=int)
    o = np.ones((2,), dtype=d.dtype)
    r = o * 5
    assert_equal(f(d), r)
    assert_equal(f(d, axis=0), r)
    assert_equal(f(d, 0), r)
    assert_equal(f(d, 0, dtype=None), r)
    assert_equal(f(d, 0, dtype='i'), r)
    assert_equal(f(d, 0, 'i'), r)
    assert_equal(f(d, 0, None), r)
    assert_equal(f(d, 0, None, out=None), r)
    assert_equal(f(d, 0, None, out=o), r)
    assert_equal(f(d, 0, None, o), r)
    assert_equal(f(d, 0, None, None), r)
    assert_equal(f(d, 0, None, None, keepdims=False), r)
    assert_equal(f(d, 0, None, None, True), r.reshape((1,) + r.shape))
    assert_equal(f(d, 0, None, None, False, 0), r)
    assert_equal(f(d, 0, None, None, False, initial=0), r)
    assert_equal(f(d, 0, None, None, False, 0, True), r)
    assert_equal(f(d, 0, None, None, False, 0, where=True), r)
    assert_equal(f(d, axis=0, dtype=None, out=None, keepdims=False), r)
    assert_equal(f(d, 0, dtype=None, out=None, keepdims=False), r)
    assert_equal(f(d, 0, None, out=None, keepdims=False), r)
    assert_equal(f(d, 0, None, out=None, keepdims=False, initial=0, where=True), r)
    assert_raises(TypeError, f)
    assert_raises(TypeError, f, d, 0, None, None, False, 0, True, 1)
    assert_raises(TypeError, f, d, 'invalid')
    assert_raises(TypeError, f, d, axis='invalid')
    assert_raises(TypeError, f, d, axis='invalid', dtype=None, keepdims=True)
    assert_raises(TypeError, f, d, 0, 'invalid')
    assert_raises(TypeError, f, d, dtype='invalid')
    assert_raises(TypeError, f, d, dtype='invalid', out=None)
    assert_raises(TypeError, f, d, 0, None, 'invalid')
    assert_raises(TypeError, f, d, out='invalid')
    assert_raises(TypeError, f, d, out='invalid', dtype=None)
    assert_raises(TypeError, f, d, 0, keepdims='invalid', dtype='invalid', out=None)
    assert_raises(TypeError, f, d, axis=0, dtype=None, invalid=0)
    assert_raises(TypeError, f, d, invalid=0)
    assert_raises(TypeError, f, d, 0, keepdims=True, invalid='invalid', out=None)
    assert_raises(TypeError, f, d, axis=0, dtype=None, keepdims=True, out=None, invalid=0)
    assert_raises(TypeError, f, d, axis=0, dtype=None, out=None, invalid=0)