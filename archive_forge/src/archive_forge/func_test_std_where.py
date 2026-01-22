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
def test_std_where(self):
    a = np.arange(25).reshape((5, 5))[::-1]
    whf = np.array([[False, True, False, True, True], [True, False, True, False, True], [True, True, False, True, False], [True, False, True, True, False], [False, True, False, True, True]])
    whp = np.array([[False], [False], [True], [True], [False]])
    _cases = [(0, True, 7.07106781 * np.ones(5)), (1, True, 1.41421356 * np.ones(5)), (0, whf, np.array([4.0824829, 8.16496581, 5.0, 7.39509973, 8.49836586])), (0, whp, 2.5 * np.ones(5))]
    for _ax, _wh, _res in _cases:
        assert_allclose(a.std(axis=_ax, where=_wh), _res)
        assert_allclose(np.std(a, axis=_ax, where=_wh), _res)
    a3d = np.arange(16).reshape((2, 2, 4))
    _wh_partial = np.array([False, True, True, False])
    _res = [[0.5, 0.5], [0.5, 0.5]]
    assert_allclose(a3d.std(axis=2, where=_wh_partial), np.array(_res))
    assert_allclose(np.std(a3d, axis=2, where=_wh_partial), np.array(_res))
    assert_allclose(a.std(axis=1, where=whf), np.std(a[whf].reshape((5, 3)), axis=1))
    assert_allclose(np.std(a, axis=1, where=whf), a[whf].reshape((5, 3)).std(axis=1))
    assert_allclose(a.std(axis=0, where=whp), np.std(a[whp[:, 0]], axis=0))
    assert_allclose(np.std(a, axis=0, where=whp), a[whp[:, 0]].std(axis=0))
    with pytest.warns(RuntimeWarning) as w:
        assert_equal(a.std(where=False), np.nan)
    with pytest.warns(RuntimeWarning) as w:
        assert_equal(np.std(a, where=False), np.nan)