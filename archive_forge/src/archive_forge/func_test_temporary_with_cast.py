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
def test_temporary_with_cast(self):
    d = np.ones(200000, dtype=np.int64)
    assert_equal((d + d + 2 ** 222).dtype, np.dtype('O'))
    r = (d + d) / 2
    assert_equal(r.dtype, np.dtype('f8'))
    r = np.true_divide(d + d, 2)
    assert_equal(r.dtype, np.dtype('f8'))
    r = (d + d) / 2.0
    assert_equal(r.dtype, np.dtype('f8'))
    r = (d + d) // 2
    assert_equal(r.dtype, np.dtype(np.int64))
    f = np.ones(100000, dtype=np.float32)
    assert_equal((f + f + f.astype(np.float64)).dtype, np.dtype('f8'))
    d = f.astype(np.float64)
    assert_equal((f + f + d).dtype, d.dtype)
    l = np.ones(100000, dtype=np.longdouble)
    assert_equal((d + d + l).dtype, l.dtype)
    for dt in (np.complex64, np.complex128, np.clongdouble):
        c = np.ones(100000, dtype=dt)
        r = abs(c * 2.0)
        assert_equal(r.dtype, np.dtype('f%d' % (c.itemsize // 2)))