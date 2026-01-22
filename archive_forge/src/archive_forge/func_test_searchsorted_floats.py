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
@pytest.mark.parametrize('a', [np.array([0, 1, np.nan], dtype=np.float16), np.array([0, 1, np.nan], dtype=np.float32), np.array([0, 1, np.nan])])
def test_searchsorted_floats(self, a):
    msg = "Test real (%s) searchsorted with nans, side='l'" % a.dtype
    b = a.searchsorted(a, side='left')
    assert_equal(b, np.arange(3), msg)
    msg = "Test real (%s) searchsorted with nans, side='r'" % a.dtype
    b = a.searchsorted(a, side='right')
    assert_equal(b, np.arange(1, 4), msg)
    a.searchsorted(v=1)
    x = np.array([0, 1, np.nan], dtype='float32')
    y = np.searchsorted(x, x[-1])
    assert_equal(y, 2)