import collections.abc
import textwrap
from io import BytesIO
from os import path
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import (
from numpy.compat import pickle
def test_recarray_views(self):
    a = np.array([(1, 'ABC'), (2, 'DEF')], dtype=[('foo', int), ('bar', 'S4')])
    b = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert_equal(np.rec.array(a).dtype.type, np.record)
    assert_equal(type(np.rec.array(a)), np.recarray)
    assert_equal(np.rec.array(b).dtype.type, np.int64)
    assert_equal(type(np.rec.array(b)), np.recarray)
    assert_equal(a.view(np.recarray).dtype.type, np.record)
    assert_equal(type(a.view(np.recarray)), np.recarray)
    assert_equal(b.view(np.recarray).dtype.type, np.int64)
    assert_equal(type(b.view(np.recarray)), np.recarray)
    r = np.rec.array(np.ones(4, dtype='f4,i4'))
    rv = r.view('f8').view('f4,i4')
    assert_equal(type(rv), np.recarray)
    assert_equal(rv.dtype.type, np.record)
    r = np.rec.array(np.ones(4, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'i4,i4')]))
    assert_equal(r['c'].dtype.type, np.record)
    assert_equal(type(r['c']), np.recarray)

    class C(np.recarray):
        pass
    c = r.view(C)
    assert_equal(type(c['c']), C)
    test_dtype = [('a', 'f4,f4'), ('b', 'V8'), ('c', ('f4', 2)), ('d', ('i8', 'i4,i4'))]
    r = np.rec.array([((1, 1), b'11111111', [1, 1], 1), ((1, 1), b'11111111', [1, 1], 1)], dtype=test_dtype)
    assert_equal(r.a.dtype.type, np.record)
    assert_equal(r.b.dtype.type, np.void)
    assert_equal(r.c.dtype.type, np.float32)
    assert_equal(r.d.dtype.type, np.int64)
    r = np.rec.array(np.ones(4, dtype='i4,i4'))
    assert_equal(r.view('f4,f4').dtype.type, np.record)
    assert_equal(r.view(('i4', 2)).dtype.type, np.int32)
    assert_equal(r.view('V8').dtype.type, np.void)
    assert_equal(r.view(('i8', 'i4,i4')).dtype.type, np.int64)
    arrs = [np.ones(4, dtype='f4,i4'), np.ones(4, dtype='f8')]
    for arr in arrs:
        rec = np.rec.array(arr)
        arr2 = rec.view(rec.dtype.fields or rec.dtype, np.ndarray)
        assert_equal(arr2.dtype.type, arr.dtype.type)
        assert_equal(type(arr2), type(arr))