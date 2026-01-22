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
def test_matmul_bool(self):
    a = np.array([[1, 0], [1, 1]], dtype=bool)
    assert np.max(a.view(np.uint8)) == 1
    b = np.matmul(a, a)
    assert np.max(b.view(np.uint8)) == 1
    rg = np.random.default_rng(np.random.PCG64(43))
    d = rg.integers(2, size=4 * 5, dtype=np.int8)
    d = d.reshape(4, 5) > 0
    out1 = np.matmul(d, d.reshape(5, 4))
    out2 = np.dot(d, d.reshape(5, 4))
    assert_equal(out1, out2)
    c = np.matmul(np.zeros((2, 0), dtype=bool), np.zeros(0, dtype=bool))
    assert not np.any(c)