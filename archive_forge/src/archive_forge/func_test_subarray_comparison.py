from __future__ import annotations
import collections.abc
import tempfile
import sys
import warnings
import operator
import io
import itertools
import functools
import ctypes
import os
import gc
import re
import weakref
import pytest
from contextlib import contextmanager
from numpy.compat import pickle
import pathlib
import builtins
from decimal import Decimal
import mmap
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.testing._private.utils import requires_memory, _no_tracing
from numpy.core.tests._locales import CommaDecimalPointLocale
from numpy.lib.recfunctions import repack_fields
from numpy.core.multiarray import _get_ndarray_c_version
from datetime import timedelta, datetime
from numpy.core._internal import _dtype_from_pep3118
from numpy.testing import IS_PYPY
def test_subarray_comparison(self):
    a = np.rec.fromrecords([([1, 2, 3], 'a', [[1, 2], [3, 4]]), ([3, 3, 3], 'b', [[0, 0], [0, 0]])], dtype=[('a', ('f4', 3)), ('b', object), ('c', ('i4', (2, 2)))])
    b = a.copy()
    assert_equal(a == b, [True, True])
    assert_equal(a != b, [False, False])
    b[1].b = 'c'
    assert_equal(a == b, [True, False])
    assert_equal(a != b, [False, True])
    for i in range(3):
        b[0].a = a[0].a
        b[0].a[i] = 5
        assert_equal(a == b, [False, False])
        assert_equal(a != b, [True, True])
    for i in range(2):
        for j in range(2):
            b = a.copy()
            b[0].c[i, j] = 10
            assert_equal(a == b, [False, True])
            assert_equal(a != b, [True, False])
    a = np.array([[(0,)], [(1,)]], dtype=[('a', 'f8')])
    b = np.array([(0,), (0,), (1,)], dtype=[('a', 'f8')])
    assert_equal(a == b, [[True, True, False], [False, False, True]])
    assert_equal(b == a, [[True, True, False], [False, False, True]])
    a = np.array([[(0,)], [(1,)]], dtype=[('a', 'f8', (1,))])
    b = np.array([(0,), (0,), (1,)], dtype=[('a', 'f8', (1,))])
    assert_equal(a == b, [[True, True, False], [False, False, True]])
    assert_equal(b == a, [[True, True, False], [False, False, True]])
    a = np.array([[([0, 0],)], [([1, 1],)]], dtype=[('a', 'f8', (2,))])
    b = np.array([([0, 0],), ([0, 1],), ([1, 1],)], dtype=[('a', 'f8', (2,))])
    assert_equal(a == b, [[True, False, False], [False, False, True]])
    assert_equal(b == a, [[True, False, False], [False, False, True]])
    a = np.array([[([0, 0],)], [([1, 1],)]], dtype=[('a', 'f8', (2,))], order='F')
    b = np.array([([0, 0],), ([0, 1],), ([1, 1],)], dtype=[('a', 'f8', (2,))])
    assert_equal(a == b, [[True, False, False], [False, False, True]])
    assert_equal(b == a, [[True, False, False], [False, False, True]])
    x = np.zeros((1,), dtype=[('a', ('f4', (1, 2))), ('b', 'i1')])
    y = np.zeros((1,), dtype=[('a', ('f4', (2,))), ('b', 'i1')])
    with pytest.raises(TypeError):
        x == y
    x = np.zeros((1,), dtype=[('a', ('f4', (2, 1))), ('b', 'i1')])
    y = np.zeros((1,), dtype=[('a', ('f4', (2,))), ('b', 'i1')])
    with pytest.raises(TypeError):
        x == y