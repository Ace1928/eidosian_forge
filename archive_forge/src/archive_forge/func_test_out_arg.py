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
def test_out_arg(self):
    a = np.ones((5, 2), dtype=float)
    b = np.array([[1, 3], [5, 7]], dtype=float)
    tgt = np.dot(a, b)
    msg = 'out positional argument'
    out = np.zeros((5, 2), dtype=float)
    self.matmul(a, b, out)
    assert_array_equal(out, tgt, err_msg=msg)
    msg = 'out keyword argument'
    out = np.zeros((5, 2), dtype=float)
    self.matmul(a, b, out=out)
    assert_array_equal(out, tgt, err_msg=msg)
    msg = 'Cannot cast ufunc .* output'
    out = np.zeros((5, 2), dtype=np.int32)
    assert_raises_regex(TypeError, msg, self.matmul, a, b, out=out)
    out = np.zeros((5, 2), dtype=np.complex128)
    c = self.matmul(a, b, out=out)
    assert_(c is out)
    with suppress_warnings() as sup:
        sup.filter(np.ComplexWarning, '')
        c = c.astype(tgt.dtype)
    assert_array_equal(c, tgt)