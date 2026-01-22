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
def test_view_discard_refcount(self):
    from numpy.core._multiarray_tests import npy_create_writebackifcopy, npy_discard
    arr = np.arange(9).reshape(3, 3).T
    orig = arr.copy()
    if HAS_REFCOUNT:
        arr_cnt = sys.getrefcount(arr)
    arr_wb = npy_create_writebackifcopy(arr)
    assert_(arr_wb.flags.writebackifcopy)
    assert_(arr_wb.base is arr)
    arr_wb[...] = -100
    npy_discard(arr_wb)
    assert_equal(arr, orig)
    assert_(arr_wb.ctypes.data != 0)
    assert_equal(arr_wb.base, None)
    if HAS_REFCOUNT:
        assert_equal(arr_cnt, sys.getrefcount(arr))
    arr_wb[...] = 100
    assert_equal(arr, orig)