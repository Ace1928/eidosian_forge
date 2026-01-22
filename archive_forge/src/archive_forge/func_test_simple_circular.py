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
def test_simple_circular(self):
    dt = np.float64
    x = np.array([1, 2, 3], dtype=dt)
    r = [np.array([0, 3, 1], dtype=dt), np.array([3, 1, 2], dtype=dt), np.array([1, 2, 3], dtype=dt), np.array([2, 3, 1], dtype=dt), np.array([3, 1, 0], dtype=dt)]
    l = _multiarray_tests.test_neighborhood_iterator_oob(x, [-1, 3], NEIGH_MODE['circular'], [-1, 1], NEIGH_MODE['zero'])
    assert_array_equal(l, r)
    x = np.array([1, 2, 3], dtype=dt)
    r = [np.array([3, 0, 0], dtype=dt), np.array([0, 0, 1], dtype=dt), np.array([0, 1, 2], dtype=dt), np.array([1, 2, 3], dtype=dt), np.array([2, 3, 0], dtype=dt)]
    l = _multiarray_tests.test_neighborhood_iterator_oob(x, [-1, 3], NEIGH_MODE['zero'], [-2, 0], NEIGH_MODE['circular'])
    assert_array_equal(l, r)
    x = np.array([1, 2, 3], dtype=dt)
    r = [np.array([0, 1, 2], dtype=dt), np.array([1, 2, 3], dtype=dt), np.array([2, 3, 0], dtype=dt), np.array([3, 0, 0], dtype=dt), np.array([0, 0, 1], dtype=dt)]
    l = _multiarray_tests.test_neighborhood_iterator_oob(x, [-1, 3], NEIGH_MODE['zero'], [0, 2], NEIGH_MODE['circular'])
    assert_array_equal(l, r)
    x = np.array([1, 2, 3], dtype=dt)
    r = [np.array([3, 0, 0, 1, 2], dtype=dt), np.array([0, 0, 1, 2, 3], dtype=dt), np.array([0, 1, 2, 3, 0], dtype=dt), np.array([1, 2, 3, 0, 0], dtype=dt), np.array([2, 3, 0, 0, 1], dtype=dt)]
    l = _multiarray_tests.test_neighborhood_iterator_oob(x, [-1, 3], NEIGH_MODE['zero'], [-2, 2], NEIGH_MODE['circular'])
    assert_array_equal(l, r)