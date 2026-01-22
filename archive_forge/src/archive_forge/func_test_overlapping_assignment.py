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
def test_overlapping_assignment(self):
    a = np.arange(4)
    a[:-1] = a[1:]
    assert_equal(a, [1, 2, 3, 3])
    a = np.arange(4)
    a[1:] = a[:-1]
    assert_equal(a, [0, 0, 1, 2])
    a = np.arange(4)
    a[:] = a[::-1]
    assert_equal(a, [3, 2, 1, 0])
    a = np.arange(6).reshape(2, 3)
    a[::-1, :] = a[:, ::-1]
    assert_equal(a, [[5, 4, 3], [2, 1, 0]])
    a = np.arange(6).reshape(2, 3)
    a[::-1, ::-1] = a[:, ::-1]
    assert_equal(a, [[3, 4, 5], [0, 1, 2]])
    a = np.arange(5)
    a[:3] = a[2:]
    assert_equal(a, [2, 3, 4, 3, 4])
    a = np.arange(5)
    a[2:] = a[:3]
    assert_equal(a, [0, 1, 0, 1, 2])
    a = np.arange(5)
    a[2::-1] = a[2:]
    assert_equal(a, [4, 3, 2, 3, 4])
    a = np.arange(5)
    a[2:] = a[2::-1]
    assert_equal(a, [0, 1, 2, 1, 0])
    a = np.arange(5)
    a[2::-1] = a[:1:-1]
    assert_equal(a, [2, 3, 4, 3, 4])
    a = np.arange(5)
    a[:1:-1] = a[2::-1]
    assert_equal(a, [0, 1, 0, 1, 2])