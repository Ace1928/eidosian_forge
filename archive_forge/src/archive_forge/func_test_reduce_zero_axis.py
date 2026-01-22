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
def test_reduce_zero_axis(self):

    def ok(f, *args, **kwargs):
        f(*args, **kwargs)

    def err(f, *args, **kwargs):
        assert_raises(ValueError, f, *args, **kwargs)

    def t(expect, func, n, m):
        expect(func, np.zeros((n, m)), axis=1)
        expect(func, np.zeros((m, n)), axis=0)
        expect(func, np.zeros((n // 2, n // 2, m)), axis=2)
        expect(func, np.zeros((n // 2, m, n // 2)), axis=1)
        expect(func, np.zeros((n, m // 2, m // 2)), axis=(1, 2))
        expect(func, np.zeros((m // 2, n, m // 2)), axis=(0, 2))
        expect(func, np.zeros((m // 3, m // 3, m // 3, n // 2, n // 2)), axis=(0, 1, 2))
        expect(func, np.zeros((10, m, n)), axis=(0, 1))
        expect(func, np.zeros((10, n, m)), axis=(0, 2))
        expect(func, np.zeros((m, 10, n)), axis=0)
        expect(func, np.zeros((10, m, n)), axis=1)
        expect(func, np.zeros((10, n, m)), axis=2)
    assert_equal(np.maximum.identity, None)
    t(ok, np.maximum.reduce, 30, 30)
    t(ok, np.maximum.reduce, 0, 30)
    t(err, np.maximum.reduce, 30, 0)
    t(err, np.maximum.reduce, 0, 0)
    err(np.maximum.reduce, [])
    np.maximum.reduce(np.zeros((0, 0)), axis=())
    t(ok, np.add.reduce, 30, 30)
    t(ok, np.add.reduce, 0, 30)
    t(ok, np.add.reduce, 30, 0)
    t(ok, np.add.reduce, 0, 0)
    np.add.reduce([])
    np.add.reduce(np.zeros((0, 0)), axis=())
    for uf in (np.maximum, np.add):
        uf.accumulate(np.zeros((30, 0)), axis=0)
        uf.accumulate(np.zeros((0, 30)), axis=0)
        uf.accumulate(np.zeros((30, 30)), axis=0)
        uf.accumulate(np.zeros((0, 0)), axis=0)