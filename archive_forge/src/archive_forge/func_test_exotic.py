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
def test_exotic(self):
    assert_array_equal(np.where(True, None, None), np.array(None))
    m = np.array([], dtype=bool).reshape(0, 3)
    b = np.array([], dtype=np.float64).reshape(0, 3)
    assert_array_equal(np.where(m, 0, b), np.array([]).reshape(0, 3))
    d = np.array([-1.34, -0.16, -0.54, -0.31, -0.08, -0.95, 0.0, 0.313, 0.547, -0.18, 0.876, 0.236, 1.969, 0.31, 0.699, 1.013, 1.267, 0.229, -1.39, 0.487])
    nan = float('NaN')
    e = np.array(['5z', '0l', nan, 'Wz', nan, nan, 'Xq', 'cs', nan, nan, 'QN', nan, nan, 'Fd', nan, nan, 'kp', nan, '36', 'i1'], dtype=object)
    m = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
    r = e[:]
    r[np.where(m)] = d[np.where(m)]
    assert_array_equal(np.where(m, d, e), r)
    r = e[:]
    r[np.where(~m)] = d[np.where(~m)]
    assert_array_equal(np.where(m, e, d), r)
    assert_array_equal(np.where(m, e, e), e)
    d = np.array([1.0, 2.0], dtype=np.float32)
    e = float('NaN')
    assert_equal(np.where(True, d, e).dtype, np.float32)
    e = float('Infinity')
    assert_equal(np.where(True, d, e).dtype, np.float32)
    e = float('-Infinity')
    assert_equal(np.where(True, d, e).dtype, np.float32)
    e = float(1e+150)
    assert_equal(np.where(True, d, e).dtype, np.float64)