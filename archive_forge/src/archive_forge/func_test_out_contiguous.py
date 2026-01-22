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
def test_out_contiguous(self):
    a = np.ones((5, 2), dtype=float)
    b = np.array([[1, 3], [5, 7]], dtype=float)
    v = np.array([1, 3], dtype=float)
    tgt = np.dot(a, b)
    tgt_mv = np.dot(a, v)
    out = np.ones((5, 2, 2), dtype=float)
    c = self.matmul(a, b, out=out[..., 0])
    assert c.base is out
    assert_array_equal(c, tgt)
    c = self.matmul(a, v, out=out[:, 0, 0])
    assert_array_equal(c, tgt_mv)
    c = self.matmul(v, a.T, out=out[:, 0, 0])
    assert_array_equal(c, tgt_mv)
    out = np.ones((10, 2), dtype=float)
    c = self.matmul(a, b, out=out[::2, :])
    assert_array_equal(c, tgt)
    out = np.ones((5, 2), dtype=float)
    c = self.matmul(b.T, a.T, out=out.T)
    assert_array_equal(out, tgt)